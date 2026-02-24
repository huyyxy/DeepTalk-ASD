import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from easydict import EasyDict
from loconet_asd.loss_multi import lossAV, lossA, lossV
from loconet_asd.model.loconet_encoder import locoencoder

import torch.distributed as dist


class Loconet(nn.Module):

    def __init__(self):
        super(Loconet, self).__init__()
        self.cfg = EasyDict({
            "MODEL": {
                "NUM_SPEAKERS": 3,
                "CLIP_LENGTH": 200,
                "AV": "speaker_temporal",
                "AV_layers": 3,
                "ADJUST_ATTENTION": 0
            }
        })
        self.model = locoencoder(self.cfg)
        self.lossAV = lossAV()
        self.lossA = lossA()
        self.lossV = lossV()

    def forward(self, audioFeature, visualFeature, labels, masks):
        b, s, t = visualFeature.shape[:3]
        visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
        labels = labels.view(b * s, *labels.shape[2:])
        masks = masks.view(b * s, *masks.shape[2:])

        audioEmbed = self.model.forward_audio_frontend(audioFeature)    # B, C, T, 4
        visualEmbed = self.model.forward_visual_frontend(visualFeature)
        audioEmbed = audioEmbed.repeat(s, 1, 1)

        audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
        outsA = self.model.forward_audio_backend(audioEmbed)
        outsV = self.model.forward_visual_backend(visualEmbed)

        labels = labels.reshape((-1))
        masks = masks.reshape((-1))
        nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels, masks)
        nlossA = self.lossA.forward(outsA, labels, masks)
        nlossV = self.lossV.forward(outsV, labels, masks)

        nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV

        num_frames = masks.sum()
        return nloss, prec, num_frames


class loconet(nn.Module):

    def __init__(self, cfg, rank=None, device=None):
        super(loconet, self).__init__()
        self.cfg = cfg
        self.rank = rank
        if rank != None:
            self.rank = rank
            self.device = device

            self.model = Loconet(cfg).to(device)
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[rank],
                                                             output_device=rank,
                                                             find_unused_parameters=False)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.SOLVER.BASE_LR)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                             step_size=1,
                                                             gamma=self.cfg.SOLVER.SCHEDULER.GAMMA)
        else:
            self.model = locoencoder(cfg).cuda()
            self.lossAV = lossAV().cuda()
            self.lossA = lossA().cuda()
            self.lossV = lossV().cuda()

        print(
            time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
            (sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def evaluate_network(self, epoch, loader):
        self.eval()
        predScores = []
        evalCsvSave = os.path.join(self.cfg.WORKSPACE, "{}_res.csv".format(epoch))
        evalOrig = self.cfg.evalOrig
        for audioFeature, visualFeature, labels, masks in tqdm.tqdm(loader):
            with torch.no_grad():
                audioFeature = audioFeature.cuda()
                visualFeature = visualFeature.cuda()
                labels = labels.cuda()
                masks = masks.cuda()
                b, s, t = visualFeature.shape[0], visualFeature.shape[1], visualFeature.shape[2]
                visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
                labels = labels.view(b * s, *labels.shape[2:])
                masks = masks.view(b * s, *masks.shape[2:])
                audioEmbed = self.model.forward_audio_frontend(audioFeature)
                visualEmbed = self.model.forward_visual_frontend(visualFeature)
                audioEmbed = audioEmbed.repeat(s, 1, 1)
                audioEmbed, visualEmbed = self.model.forward_cross_attention(
                    audioEmbed, visualEmbed)
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
                labels = labels.reshape((-1))
                masks = masks.reshape((-1))
                outsAV = outsAV.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
                labels = labels.view(b, s, t)[:, 0, :].view(b * t).cuda()
                masks = masks.view(b, s, t)[:, 0, :].view(b * t)
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels, masks)
                predScore = predScore[:, 1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series(['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1, inplace=True)
        evalRes.drop(['instance_id'], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s " % (evalOrig,
                                                                                      evalCsvSave)
        mAP = float(
            str(subprocess.run(cmd, shell=True, capture_output=True).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location='cpu')
        if self.rank != None:
            info = self.load_state_dict(loadedState)
        else:
            new_state = {}

            for k, v in loadedState.items():
                new_state[k.replace("model.module.", "")] = v
            info = self.load_state_dict(new_state, strict=False)
        print(info)

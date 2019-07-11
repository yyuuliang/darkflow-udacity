import os
import time
import numpy as np
import tensorflow as tf
import pickle
from utils.udacity_voc_csv import udacity_voc_csv

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        # fetches = [self.train_op, loss_op] 
        fetches = [self.train_op, loss_op, self.summary_op] 
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)


def predict(self):
    inp_path = self.FLAGS.test
    all_inp_ = os.listdir(inp_path)
    all_inp_ = [i for i in all_inp_ if self.framework.is_inp(i)]
    if not all_inp_:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inp_))

    Q = 0
    TAP = 0.0
    testdumps = []
    if self.FLAGS.loglevel > 0:
        testdumps = udacity_voc_csv('', self.meta['labels'],False,True)
        # print(testdumps)

    for j in range(len(all_inp_) // batch):
        Q = Q +1
        inp_feed = list(); new_all = list()
        all_inp = all_inp_[j*batch: (j*batch+batch)]
        for inp in all_inp:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        all_inp = new_all

        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()


        PR = np.zeros((batch,2))
        for i, prediction in enumerate(out):
            p,r = self.framework.postprocess(prediction,
                os.path.join(inp_path, all_inp[i]), True,testdumps)
            PR[i]=[p,r]
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
            
        if self.FLAGS.loglevel > 0:
            # calculate AP
            # ref: https://sanchom.wordpress.com/tag/average-precision/
            AP = 0.0
            prevR = 0.0
            ind = np.argsort( PR[:,1] ); 
            PR = PR[ind]
            for i in range(0,batch-1):
                AP = AP + PR[i][0]*(PR[i][1]-prevR)
                prevR = PR[i][1]
            AP = AP/10000
            TAP = TAP + AP
            if self.FLAGS.loglevel > 1:
                print('AP:{0}'.format(AP))
    # mAP is AP/Q
    # Q is the number of query, if we do "test" once, Q = 1, if we do "test" twice, Q = 2
    # Here, Q = total images / batch size
    if self.FLAGS.loglevel > 0:
        print('mAP:{0} at threshold: 0.5'.format(TAP/Q))
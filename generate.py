import os
import time
import logging
import fire

import note_seq
from magenta.models.music_vae import music_vae_generate
from magenta.models.music_vae import TrainedModel
import tensorflow.compat.v1 as tf


class DrumGenerator():
    def __init__(self, config, checkpoint_file, mode, num_outputs, output_dir):
        self.FLAGS = music_vae_generate.FLAGS
        self.FLAGS.checkpoint_file = checkpoint_file
        self.FLAGS.mode = mode
        self.FLAGS.num_outputs = num_outputs
        self.FLAGS.output_dir = output_dir
        
        config_map = music_vae_generate.configs.CONFIG_MAP
        self.config = config_map[config]
        
        
    def sample(self):
        date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
            
        logging.info('Loading model...')
        checkpoint_dir_or_path = os.path.expanduser(self.FLAGS.checkpoint_file)
        model = TrainedModel(
            self.config, 
            batch_size=min(self.FLAGS.max_batch_size, self.FLAGS.num_outputs),
            checkpoint_dir_or_path=checkpoint_dir_or_path)
            
        logging.info('Sampling...')
        results = model.sample(
            n=self.FLAGS.num_outputs,
            length=self.config.hparams.max_seq_len)
        
        basename = os.path.join(
            self.FLAGS.output_dir,
            '%s_%s_%s-*-of-%03d.mid' %
            (self.FLAGS.config, self.FLAGS.mode, date_and_time, self.FLAGS.num_outputs))
        logging.info('Outputting %d files as `%s`...', self.FLAGS.num_outputs, basename)
        for i, ns in enumerate(results):
            note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))
            
        logging.info('Done.')
        
        
if __name__ == "__main__":
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(music_vae_generate.FLAGS.log)
    fire.Fire(DrumGenerator)


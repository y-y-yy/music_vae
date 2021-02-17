import os
import fire

from magenta.models.music_vae import music_vae_train
from magenta.models.music_vae import configs
from magenta.models.music_vae import data
import tensorflow.compat.v1 as tf


class DrumLearner():
    def __init__(self, config, mode, run_dir, examples_path):
        self.FLAGS = music_vae_train.FLAGS
        self.FLAGS.mode = mode
        self.FLAGS.run_dir = run_dir
        self.FLAGS.examples_path = examples_path
        
        config_map = music_vae_train.configs.CONFIG_MAP
        self.config = config_map[config]
        
        
    def run(self):
        ref_dir = os.path.expanduser(self.FLAGS.run_dir)
        train_dir = os.path.join(ref_dir, 'train')
        
        config_update_map = {}
        if self.FLAGS.examples_path:
            config_update_map['%s_examples_path' % self.FLAGS.mode] = os.path.expanduser(self.FLAGS.examples_path)
        self.config = configs.update_config(self.config, config_update_map)
        
        def dataset_fn():
            is_training = True if self.FLAGS.mode == 'train' else False
            
            return data.get_dataset(
                self.config, 
                is_training = is_training)
                
        return ref_dir, train_dir, dataset_fn
        
        
    def train(self, num_steps):
        self.FLAGS.num_steps = num_steps
        
        ref_dir, train_dir, dataset_fn = self.run()

        music_vae_train.train(
            train_dir,
            config = self.config,
            num_steps = self.FLAGS.num_steps,
            dataset_fn = dataset_fn)
        
        
    def evaluate(self, eval_num_batches):
        self.FLAGS.eval_num_batches = eval_num_batches
        
        ref_dir, train_dir, dataset_fn = self.run()
        eval_dir = os.path.join(ref_dir, 'eval')
        
        music_vae_train.evaluate(
            train_dir,
            eval_dir,
            config = self.config,
            num_batches = self.FLAGS.eval_num_batches,
            dataset_fn = dataset_fn)

        
if __name__ == "__main__":
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(music_vae_train.FLAGS.log)
    fire.Fire(DrumLearner)


import os
import fire
import tarfile

from magenta.scripts import convert_dir_to_note_sequences as dir_to_ns
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'ckpt_step', None,
    'Target step to convert checkpoint files into a tar file')


class Converter():
    def __init__(self):
        self.path = os.getcwd()

        
    def midi_to_tfr(self):
        dir_to_ns.convert_directory(root_dir=self.path + '/groove-v1.0.0-midionly',
                                    output_file=self.path + '/groove.tfrecord',
                                    recursive=True)

        
    def ckpt_to_tar(self, ckpt_step):
        files = ['groovae_4bar/train/model.ckpt-%d.data-00000-of-00001'%(ckpt_step),
                 'groovae_4bar/train/model.ckpt-%d.index'%(ckpt_step),
                 'groovae_4bar/train/model.ckpt-%d.meta'%(ckpt_step)]
        
        tar = tarfile.open('groovae_4bar.tar', 'x:gz')
        for file in files:
            tar.add(file)
        tar.close()

        
if __name__ == "__main__":
    fire.Fire(Converter)

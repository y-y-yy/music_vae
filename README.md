# music_vae

Magenta MusicVAE를 이용하여 4마디의 드럼 샘플을 생성한다.

실행 순서는 아래와 같다.

1. 드럼 미디 파일 [Groove Dataset](https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip) 다운로드 > 압축 해제
2. 미디 파일이 포함된 폴더를 tfrecord 파일로 변환 (groove-v1.0.0-midionly > groove.tfrecord)
```bash
python convert.py midi_to_tfr
```
3. groovae_4bar 이용하여 드럼 학습
```bash
python train.py train --config=groovae_4bar --run_dir='groovae_4bar' --mode=train --examples_path='groove.tfrecord' --num_steps=4000
```
4. 모델 평가
```bash
python train.py evaluate --config=groovae_4bar --run_dir='groovae_4bar' --mode=eval --examples_path='groove.tfrecord' --eval_num_batches=10
tensorboard --logdir='./groovae_4bar/eval'
```
5. 모델의 checkpoint 파일을 tarfile로 압축 (ckpt files > groovae_4bar.tar)
```bash
python convert.py ckpt_to_tar --ckpt_step=3557
```
6. 샘플 생성
```bash
python generate.py sample --config=groovae_4bar --checkpoint_file='groovae_4bar.tar' --mode=sample --num_outputs=5 --output_dir='./'
```

참고로 groove.tfrecord와 groovae_4bar.tar는 파일 용량 문제로 업로드가 불가하였다.

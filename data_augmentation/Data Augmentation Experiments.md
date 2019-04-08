# Data Augmentation

[TOC]

## Baseline

```yaml
augmentations:
  gamma: 1.25                                     #[gamma varied in 1 to 1+x]
  hue: 0.25                                       #[hue varied in -x to x]
  FlipChannels: 0.5																#flip channels with chance p
  scale: [384, 384]
  rcrop: [256, 256]                            #[crop of size (h,w)]
  rotate: 45                                   #[rotate -d to d degrees]
  hflip: 0                                     #[flip horizontally with chance p]
  vflip: 0                                     #[flip vertically with chance p]
```

- **Mean IoU**: 0.5142
- **Mean Acc.**: 0.6407

## Brigitness

```yaml
augmentations:
  gamma: 1.25                                  #[gamma varied in 1 to 1+x]
  hue: 0.25                                    #[hue varied in -x to x]

  brightness: 0.5                              #[brightness varied in 1-x to 1+x]

  FlipChannels: 0.5
  scale: [384, 384]
  rcrop: [256, 256]                            #[crop of size (h,w)]
  rotate: 45                                   #[rotate -d to d degrees]
  hflip: 0                                     #[flip horizontally with chance p]
  vflip: 0                                     #[flip vertically with chance p]
```

- **Mean IoU:** 0.5132(低于Baseline 0.1%)
- **Mean Acc.**: 0.6423(高于Baseline 0.2%)

## Contrast

```yaml
augmentations:
  gamma: 1.25                                     #[gamma varied in 1 to 1+x]
  hue: 0.25                                       #[hue varied in -x to x]
  
  contrast: 0.25                                  #[contrast varied in 1-x to 1+x]
  
  FlipChannels: 0.5
  scale: [384, 384]
  rcrop: [256, 256]                                #[crop of size (h,w)]
  rotate: 45                                    #[rotate -d to d degrees]
  vflip: 0                                     #[flip vertically with chance p]
```
- **Mean IoU:** 0.5075(低于Baseline 0.7%)
- **Mean Acc.**: 0.6416(高于Baseline 0.1%)

## Random Gaussian Blur

```yaml
augmentations:
  gamma: 1.25                                     #[gamma varied in 1 to 1+x]
  hue: 0.25                                       #[hue varied in -x to x]
  FlipChannels: 0.5

  RandomGaussianBlur: 1

  scale: [384, 384]
  rcrop: [256, 256]                                #[crop of size (h,w)]
  rotate: 45                                    #[rotate -d to d degrees]
  hflip: 0                                   #[flip horizontally with chance p]
  vflip: 0                                     #[flip vertically with chance p]
```

- **Mean IoU:** 0.5166(高于Baseline 0.2%)
- **Mean Acc.**: 0.6458(高于Baseline 0.5%)

## Random Horizontally Flip

```yaml
augmentations:
  gamma: 1.25                                     #[gamma varied in 1 to 1+x]
  hue: 0.25                                       #[hue varied in -x to x]
  FlipChannels: 0.5
  scale: [384, 384]
  rcrop: [256, 256]                                #[crop of size (h,w)]
  rotate: 45                                    #[rotate -d to d degrees]

  hflip: 0.5                                     #[flip horizontally with chance p]

  vflip: 0                                     #[flip vertically with chance p]
```

- **Mean IoU:** 0.5161(高于Baseline 0.2%)
- **Mean Acc.**: 0.6444(高于Baseline 0.4%)

## Dropout || Grayscale || Gaussian Noise

```python
self.joint_seq = iaa.Sequential([
    iaa.SomeOf((0, 2), [
        sometimes(iaa.OneOf([
            iaa.EdgeDetect(alpha=(0, 0.7)),
            iaa.DirectedEdgeDetect(
                alpha=(0, 0.7), direction=(0.0, 1.0)
            ),
        ])),
        iaa.AdditiveGaussianNoise(
            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
        ),
        iaa.OneOf([
            iaa.Dropout((0.01, 0.075), per_channel=0.3),
            iaa.CoarseDropout(
                (0.03, 0.05), size_percent=(0.02, 0.05),
                per_channel=0.2
            ),
        ]),
        iaa.Invert(0.05, per_channel=True),  # invert color channels
        iaa.Grayscale(alpha=(0.0, 1.0)),
    ],
               random_order=True),
], random_order=True)
```

- **Mean IoU:** 0.5247(高于Baseline 1%)
- **Mean Acc.**: 0.6571(高于Baseline 1.7%)

## Multiscale

```yaml
augmentations:
  MultiScale: 0.5																	#do multi-scale with chance p
  
  gamma: 1.25                                     #[gamma varied in 1 to 1+x]
  hue: 0.25                                       #[hue varied in -x to x]
  FlipChannels: 0.5
  scale: [384, 384]
  rcrop: [256, 256]                                #[crop of size (h,w)]
  rotate: 45                                    #[rotate -d to d degrees]
  vflip: 0                                     #[flip vertically with chance p]
```

- **Mean IoU:** 0.5217(高于Baseline 0.8%)
- **Mean Acc.**: 0.6471(高于Baseline 0.7%)

## Saturation

```yaml
augmentations:
  gamma: 1.25                                     #[gamma varied in 1 to 1+x]
  hue: 0.25                                       #[hue varied in -x to x]

  saturation: 0.5                                #[saturation varied in 1-x to 1+x]

  FlipChannels: 0.5
  scale: [384, 384]
  rcrop: [256, 256]                                #[crop of size (h,w)]
  rotate: 45                                    #[rotate -d to d degrees]
  vflip: 0                                     #[flip vertically with chance p]
```

- **Mean IoU:** 0.533(高于Baseline 2%)
- **Mean Acc.**: 0.6769(高于Baseline 3.6%)
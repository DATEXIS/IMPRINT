"""
To run interesting experiments covering imprinting task_acc vs. number of
proxies, we here define interesting multimodal tasks.
For once, it's simply random class selections from ImageNet, but we also
want to do some combined (class remapping) tasks, which are certainly non-unimodal
and presumably more difficult (e.g., class 0: collie and coral reef, class 1:
German shepherd and volcano; notice the "overlap").
One goal here is to get graphs which show a correlation between peak/saturation
of k-means k value and the number of classes combined in a class.

Here, we first define some. We will run our experiments only with these:
- 50 random (but fixed (see seed below)) classes (for random imprinting)
    - to be able to do 5x10 imprinting tasks of short, odd, even, and long on
      Imagenet data
- 100 random (but fixed (see seed below)) classes
    - to be able to do imprinting and NC measurement of random class remappings
      (1 in 1, 2 in 1, ..., 10 in 1), randomly 10 times
"""

import os
import random
import pandas as pd

SEED = 42  # Keep this fixed -- *DO NOT CHANGE*
random.seed(SEED)
fifty_fixed_random_ints = random.sample(range(1000), 50)
# [654, 114, 25, 759, 281, 250, 228, 142, 754, 104, 692, 758, 913, 558, 89, 604, 432, 32, 30, 95, 223, 238, 517, 616, 27, 574, 203, 733, 665, 718, 429, 225, 459, 603, 284, 828, 890, 6, 777, 825, 163, 714, 348, 159, 220, 980, 781, 344, 94, 389]

SEED = 420  # Keep this fixed -- *DO NOT CHANGE*
random.seed(SEED)
hundred_fixed_random_ints = random.sample(range(1000), 100)
# [26, 689, 800, 373, 279, 412, 97, 782, 679, 694, 672, 116, 30, 534, 333, 496, 665, 18, 992, 793, 639, 130, 559, 339, 441, 274, 126, 912, 940, 344, 278, 847, 149, 698, 109, 267, 503, 188, 493, 683, 104, 253, 507, 9, 403, 624, 815, 84, 528, 32, 693, 956, 734, 566, 923, 882, 518, 214, 594, 151, 306, 143, 540, 8, 823, 969, 757, 122, 430, 723, 24, 17, 699, 199, 808, 532, 192, 402, 677, 478, 684, 210, 946, 447, 491, 775, 129, 47, 642, 564, 463, 462, 189, 727, 473, 479, 213, 3, 885, 681]


IMAGENET_CLASS_FOCUS = list(set(fifty_fixed_random_ints + hundred_fixed_random_ints))

### Define the tasks we are interested in
## Random tasks
RANDOM_TASKS = []
for _i in range(5):
    RANDOM_TASKS.append([fifty_fixed_random_ints[_i * 10 : (_i + 1) * 10]])

## Random class remappings
RANDOM_CLASS_REMAPPINGS = {i: [] for i in range(1, 11)}
for _ in range(10):
    random.shuffle(hundred_fixed_random_ints)

    for i in range(1, 11):  # mapping i classes into 1
        d = {}
        # Then fill with the selected classes
        for j in range(10):
            for k in range(i):
                d[hundred_fixed_random_ints[j * 10 + k]] = j

        RANDOM_CLASS_REMAPPINGS[i].append(d)
# Checks
for i in range(1, 11):
    assert len(RANDOM_CLASS_REMAPPINGS[i]) == 10
    for d in RANDOM_CLASS_REMAPPINGS[i]:
        assert len(set(d.values())) == 10

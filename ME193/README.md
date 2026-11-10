# Ping-pong returns

**The task:** can you hit the ball with a paddle just so, so that the ball's apex is 1.2 m from the ground? The paddle rests at 0.5 m off the ground.

[`ping_pong_paddle.py`](ping_pong_paddle.py) is a MuJoCo simulation of a red ball and a paddle. The paddle learns from a reward how to keep the ball peaking at 1.2 m while keeping it over the middle. [`demo.mp4`](demo.mp4) shows the paddle before, during and after learning, with the reward on screen.

## Run it

On a laptop (no GPU needed, about 80 s):

```
pip install mujoco imageio imageio-ffmpeg
python ping_pong_paddle.py
```

On the Tufts cluster, from `Public stuff/mujoco-hpc/`:

```
python3 runIt.py ../../ME193/ping_pong_paddle.py
```

## How the paddle works

Each rally starts with the ball at a random spot, drifting sideways, from a random height. The ball is soft (restitution about 0.5), so a still paddle would let it die; the paddle has to swing.

Before every hit, the paddle predicts where, when and how fast the ball will arrive at 0.5 m. Then it:

1. **slides** under that spot, so it moves with the ball;
2. **tilts** its face by `c_pos × offset + c_vel × sideways speed` to steer the ball back to the middle;
3. **pulls back** and **swings up** through 0.5 m at `u0 + k × (arrival speed − 3.64 m/s)`, where 3.64 m/s is how fast a ball that peaked at exactly 1.2 m arrives.

The four numbers `u0`, `k`, `c_pos` and `c_vel` are what it learns.

## The reward

For each bounce: `−(apex − 1.2 m)² − 0.5 × (distance from the middle)²`. A missed ball counts as an apex of 0 m. Higher is better; 0 is perfect.

The learning method is the cross-entropy method. It tries 32 sets of the four numbers, keeps the best 6, fits a new guess to them, and repeats 9 times.

## Results

Same rally (dropped off-center, drifting sideways) at three stages:

| Stage | Apex heights (m) | Off-center (cm) |
|---|---|---|
| Before learning | 0.73 0.58 0.54 0.53 0.52 0.52 | 1 7 11 13 14 14 |
| Learning (iteration 2) | 0.94 0.82 0.83 0.61 0.72 0.58 | 2 0 0 1 1 1 |
| After learning | 1.20 1.17 1.18 1.20 1.20 1.16 | 3 1 2 1 1 0 |

Learned policy:

- swing speed = 1.03 − 0.37 × (arrival speed − 3.64) m/s
- tilt = 0.43 × offset + 0.09 × sideways speed (rad)

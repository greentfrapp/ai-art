from gym.envs.registration import register

register(
    id='PaintSvg-v0',
    entry_point='paint_svg.envs:PaintSvg',
)

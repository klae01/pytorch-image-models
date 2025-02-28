""" Scheduler Factory
Hacked together by / Copyright 2021 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .multistep_lr import MultiStepLRScheduler
from .plateau_lr import PlateauLRScheduler
from .poly_lr import PolyLRScheduler
from .step_lr import StepLRScheduler
from .tanh_lr import TanhLRScheduler


def create_scheduler(args, optimizer):
    num_steps = args.steps

    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_steps for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_steps
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(args, 'lr_noise_pct', 0.67),
        noise_std=getattr(args, 'lr_noise_std', 1.),
        noise_seed=getattr(args, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(args, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(args, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(args, 'lr_cycle_limit', 1),
    )

    lr_scheduler = None
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        num_steps = lr_scheduler.get_cycle_length() + args.cooldown_steps
    elif args.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            t_in_steps=True,
            **cycle_args,
            **noise_args,
        )
        num_steps = lr_scheduler.get_cycle_length() + args.cooldown_steps
    elif args.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_steps,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            **noise_args,
        )
    elif args.sched == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=args.decay_milestones,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            **noise_args,
        )
    elif args.sched == 'plateau':
        mode = 'min' if 'loss' in getattr(args, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=args.decay_rate,
            patience_t=args.patience_steps,
            lr_min=args.min_lr,
            mode=mode,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            cooldown_t=0,
            **noise_args,
        )
    elif args.sched == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=args.decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=num_steps,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        num_steps = lr_scheduler.get_cycle_length() + args.cooldown_steps

    return lr_scheduler, num_steps

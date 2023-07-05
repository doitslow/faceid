import logging

__all__ = ['rank0_only', 'starit']

def rank0_only(func):
    def decorated(instance, *args, **kwargs):
        if instance.ddp and not instance.rank0:
            pass
        else:
            func(instance, *args, **kwargs)

    return decorated

def starit(func):
    def decorated(*args, **kwargs):
        logging.info("*" * 80)
        func(*args, **kwargs)
        logging.info("*" * 80)

    return decorated
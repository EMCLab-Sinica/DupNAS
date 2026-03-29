def sample_blk_choice_str(sample):
    sn_blk_choice_str = "<" + ','.join([str(c) for c in sample]) + ">"  
    return sn_blk_choice_str
    


def debug_get_net_info(parent):
    info = {}

    acc, config, efficiency = parent #, imc

    # latency?
    info['config'] = config

    return info

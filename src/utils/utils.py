import numpy as np


def get_result_name(args: dict) -> str:
    
    res_str = ''
    for key, value in args.items():
        if key == 'T':
            continue
        if key == 'dt':
            continue
        if key == 'Ib':
            try:
                Ib = np.array(value)
            except SyntaxError:
                Ib = value
            value = f"{Ib[0][0]}_{Ib[1][1]}_{Ib[2][2]}"
            continue
        if key == 'm':
            continue
        if key == 'Ib_model':
            continue
        if key == 'm_model':
            continue
        if key in ('tau_max', 'force_max', 'u_min', 'u_max'):
            continue
        if key == 'save_root_path':
            continue
        if key == 'extra_info':
            continue
        res_str += f"{value}__"

    res_str = res_str[:-2]
    if args['extra_info']:
        res_str = f"{res_str}__{args['extra_info']}"

    return res_str
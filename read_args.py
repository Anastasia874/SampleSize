def read_args(sys_argv, default):
    msg = '';
    args = {}
    try:    
        filename = 'uploads/' + sys_argv[1]
        f = open(filename, 'r')
    except:    
        msg += 'Failed to open file ' + filename + '\n'
    try:
        args.method = sys_argv[2]
    except:
        msg += 'No method specified. Using default.\n'
        args.method = default.method
    try: 
        alpha = float(sys_argv[3])
    except:
        msg += 'Unable to convert alpha to float. Alpha set to ' + str(default.alpha)+'.\n'
        alpha = default.alpha
    try: 
        power = float(sys_argv[4])
    except:
        msg += 'Unable to convert alpha to float. Power set to ' +str(default.power)+'.\n'
        power = default.power
    try: 
        delta = float(sys_argv[5])
    except:
        msg += 'Unable to convert alpha to float. Power set to 0.05' +str(default.delta)+'\n'
        delpa = default.delta    
        
    return (msg, method, alpha, power, delta)

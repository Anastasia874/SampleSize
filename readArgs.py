def read_args(sys_argv, default):
    msg = '';
    args = {}
    
    try:
        args['method'] = sys_argv[2]
    except:
        msg += 'No method specified. Using default.\n'
        args['method'] = default['method']
    try: 
        args['alpha'] = float(sys_argv[3])
    except:
        msg += 'Unable to convert alpha to float. Alpha set to ' + str(default['alpha'])+'.\n'
        args['alpha'] = default['alpha']
    try: 
        args['power'] = float(sys_argv[4])
    except:
        msg += 'Unable to convert alpha to float. Power set to ' +str(default['power'])+'.\n'
        args['power'] = default['power']
    try: 
        args['delta'] = float(sys_argv[5])
    except:
        msg += 'Unable to convert alpha to float. Power set to 0.05' +str(default['delta'])+'\n'
        args['delta'] = default['delta']   
        
    return (msg, args)

def read_data(text_data, delimeter):
    data = []
    for row in text_data:
        data.append([])
        row = row.split(delimeter)
        for cell in row:
            data[-1].append(float(cell))
           
    return data


    
    

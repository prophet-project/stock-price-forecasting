import json

def save_metrict(model, results, outfile):
    metrics = {}

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))   
        metrics[name] = value

    
    json.dump(metrics, outfile, indent=4)
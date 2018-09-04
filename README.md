# app-analytics

Demand Forecast App Usage based on the past access data from Google Analytics

# Start Jupyter Notebook

```
scripts/jupyter.sh 
```

Place Google Analytics Access File in data/input/ - WIP. Automating Steps to pull the Report

If running Jupyter on remote server - start the following in local host

```
scripts/jupyter_tunnel_to_remote.sh <ssh_user_name@ip_address>
```

Access the Notebook at 

```
http://localhost:8888
```

Before Check In to Git 

```
python3 scripts/strip_output.py notebook/demand_forecast_by_ga.ipynb
```

# Best Practices in Managing Jupyter Notebooks
Seperate Visualization code with following check to enable seamless headless execution
```
shell = "ZMQInteractiveShell"
IN_JUPYTER = 'get_ipython' in globals() and \
            get_ipython().__class__.__name__ == shell

if IN_JUPYTER:
  visualize_data(df)
```


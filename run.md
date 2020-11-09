Train

```
python3 src/main.py --config=qmix_beta --env-config=sisl with env_args.env_name=waterworld
python3 src/main.py --config=qmix_beta --env-config=sisl with env_args.env_name=pursuit
python3 src/main.py --config=qmix_beta --env-config=sisl with env_args.env_name=multiwalker
```

Eval:

```
python3 src/main.py --config=qmix --save_replay=True --env-config=sisl with env_args.env_name=waterworld
```

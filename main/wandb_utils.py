from collections import defaultdict

import numpy as np
import wandb


def delete_artifacts(wand_api):
    for run in wand_api.runs():
        logged_artifact = run.logged_artifacts()

        if len(logged_artifact) == 0:
            continue

        files = sorted([f for f in run.logged_artifacts()], key=lambda f: f.updated_at)

        print("Total files:", len(files))
        print("Last file:", files[-1].name)
        print("Last file date:", files[-1].updated_at)

        for f in files:
            file_name = f.name
            if "best" in file_name:
                print(f"Not deleting best artifact '{file_name}'")
                continue

            if "latest" or "v0" in file_name:
                a = api.artifact(f"{project}/{file_name}")
                print(f"Deleting {file_name}")
                deleted = a.delete(delete_aliases=True)
                int(f"Deleted '{file_name}': {deleted}")


def update_config(wandb_api):
    # proto_maml_ids = []
    # for run_id in proto_maml_ids:
    #     run = wandb_api.run(f"rahelhabacker/meta-gnn/{run_id}")
    #     run.config["model_params/model"] = "proto-maml"
    #     run.update()

    maml_ids = ["2j23fecw", "3lqoafr9", "nksr4bq3"]
    for run_id in maml_ids:
        run = wandb_api.run(f"rahelhabacker/meta-gnn/{run_id}")
        run.config["model_params/model"] = "maml"
        run.update()


def get_mean_stds_gossipcop(wandb_api):
    f1_fakes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    filters = {
        "$and": [{
            'created_at': {
                "$gt": '2022-07-29T##'
            }
        }]
    }

    for run in wandb_api.runs(filters=filters):
        if not run.config['suffix'] in ['best_eval_gossipcop', 'best_eval_gosispcop']:
            continue

        model_name = get_model_name(run.config)
        if model_name is None:
            continue

        k_shot = run.config['k_shot']
        seed = run.config['seed']

        if seed in f1_fakes[model_name][k_shot]:
            print(f"Double Seed {seed}; continuing!")
            continue

        f1_fakes[model_name][k_shot][seed] = run.summary['test/f1_fake']

    print("Dict creation done.")

    f1_fakes_np = np.zeros((3, 4, 3))

    for i, (model, model_values) in enumerate(sorted(f1_fakes.items())):
        for j, (k_shot, shot_values) in enumerate(sorted(model_values.items())):
            for k, seed in enumerate(shot_values.values()):
                f1_fakes_np[i, j, k] = seed

    print("Numpy array creation done.")

    stds = np.std(f1_fakes_np, axis=2)
    means = np.mean(f1_fakes_np, axis=2)

    print(f"\nStandard deviations:\n{np.round(stds, 4)}\nMeans:\n{np.round(means, 2)}\n")

    return means


def get_model_name(config):
    try:
        return config['model_params']['model']
    except KeyError:
        try:
            return config['model_params.model']
        except KeyError:
            try:
                return config['model_params/model']
            except KeyError:
                return None


def get_mean_stds_twitter(wandb_api):
    f1_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

    filters = {
        "$and": [{
            'created_at': {
                "$gt": '2022-07-28T##'
            }
        }]
    }

    for run in wandb_api.runs(filters=filters):
        if not run.config['suffix'] in ['best_eval_twitter']:
            continue

        model_name = get_model_name(run.config)
        if model_name is None:
            continue

        k_shot = run.config['k_shot']
        seed = run.config['seed']

        if seed in f1_scores[model_name][k_shot]:
            print(f"Double Seed {seed}; continuing!")
            continue

        if 'test/f1_racism' not in run.summary or 'test/f1_sexism' not in run.summary or \
                'test/f1_none' not in run.summary:
            continue

        f1_scores[model_name][k_shot][seed]['racism'] = run.summary['test/f1_racism']
        f1_scores[model_name][k_shot][seed]['sexism'] = run.summary['test/f1_sexism']
        f1_scores[model_name][k_shot][seed]['none'] = run.summary['test/f1_none']

    print("Dict creation done.")

    f1_scores_np = np.zeros((3, 4, 3, 3))

    for i, (model, model_values) in enumerate(sorted(f1_scores.items())):
        for j, (k_shot, shot_values) in enumerate(sorted(model_values.items())):
            for k, seed in enumerate(shot_values.values()):
                for l, target_class in enumerate(seed.values()):
                    f1_scores_np[i, j, k, l] = target_class

    print("Numpy array creation done.")

    stds = np.std(f1_scores_np, axis=2)
    means = np.mean(f1_scores_np, axis=2)

    print(f"Racism:\nStd:\n{np.round(stds[:, :, 0], 4)}\nMeans:\n{np.round(means[:, :, 0], 2)}\n\n"
          f"Sexism:\nStd:\n{np.round(stds[:, :, 1], 4)}\nMeans:\n{np.round(means[:, :, 1], 2)}\n\n"
          f"None:\nStd:\n{np.round(stds[:, :, 2], 4)}\nMeans:\n{np.round(means[:, :, 2], 2)}\n")

    return means


if __name__ == '__main__':
    key = "cde3393a5821845320af99419cf2f90c0bf094e6"
    project = "meta-gnn"
    entity = "rahelhabacker"

    wandb.login(key=key)

    api = wandb.Api(overrides={"project": project, "entity": entity}, timeout=19)

    g_means = get_mean_stds_gossipcop(api)

    t_means = get_mean_stds_twitter(api)

    # update_config(api)

    print("foo.")

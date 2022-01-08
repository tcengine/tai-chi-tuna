__all__ = ["ParamWizard"]

import pandas as pd
from typing import Dict, Any
import torch

class ParamWizard:
    """
    A powerful handler that can manage discriminative configurations
    """

    def __init__(self, model):
        self.model = model
        self.param_dict = dict(model.named_parameters())
        self.param_level = dict((k, len(list(k.split('.'))))
                                for k in self.param_dict.keys())
        self.create_names_df()

    def create_names_df(self,):
        self.names = list(self.param_dict.keys())
        self.names_df = pd.DataFrame(
            dict(
                named=self.param_level.keys(),
                level=self.param_level.values(),
            )
        )

    def set_configure_optimizers(self, phase):
        """
        Assign the configure_optimizers to the model
        """
        def configure_optimizers(pl_cls,):
            if 'param_groups' not in phase.config:
                param_groups = pl_cls.parameters()
            elif len(phase['param_groups'])==0:
                param_groups = pl_cls.parameters()
            else:
                param_groups = self.create_param_groups(phase['param_groups'])
            opt = torch.optim.Adam(param_groups, lr=1e-4)
            pl_cls.optimizer = opt
            return opt
        
        self.model.__class__.configure_optimizers = configure_optimizers

    def find_prefix(self, prefix: str):
        """
        Find all the parameters with a given prefix
        """
        return list(self.names_df[self.names_df.named.str.startswith(prefix)]['named'])

    def find_next_level(self, prefix: str):
        """
        Find all the sub level with given prefix
        """
        prefix_level = len(prefix.split('.')) if prefix != "" else 0
        df = self.names_df[self.names_df.named.str.startswith(prefix)]
        if len(df) <= 1:
            return None
        sub_col = df['named'].apply(
            lambda x: x.split('.')[prefix_level])
        col_ct = sub_col.value_counts()
        return col_ct

    def find_kw(self, prefix: str):
        df = self.names_df[self.names_df.named.str.startswith(prefix)]

        kw_df = pd.DataFrame(
            df['named'].apply(lambda x: x.split('.')).explode().value_counts()
        ).reset_index()

        kw_df = kw_df.rename(columns={"index": "kw"})
        if prefix == "":
            return kw_df
        kw_df = kw_df[~kw_df.kw.isin(prefix.split('.'))].reset_index(drop=True)
        return kw_df

    def __len__(self):
        return len(self.param_dict)

    def __getitem__(self, key):
        return self.param_dict[key]

    def create_conf_dict(self, conf):
        if "lr" in conf:
            conf['lr'] = float(conf['lr'])
        if "freeze" in conf:
            del conf['freeze']
        return conf

    def grouping(self, prefix_dict: Dict[str, Dict[str, Any]]):
        """
        Create ```param_groups```
            that can be used for initializing the optimizer
        """
        prefix_list = []
        kw_list = []

        for key in prefix_dict.keys():
            prefix, kw = key.split("|")

            prefix_list.append(prefix)
            kw_list.append(kw)

        group_df = pd.DataFrame(
            {"prefix": prefix_list, "kw": kw_list, "conf": list(prefix_dict.values())})
        group_df['level'] = group_df.prefix.apply(lambda x: len(x.split('.')))
        group_df = group_df.sort_values(
            by=['level', "kw", 'prefix'], ascending=True
        ).reset_index(drop=True)

        group_df['sn'] = list(range(len(group_df)))

        mapper = self.names_df.copy()
        mapper['group'] = "*"

        for idx, group in group_df.iterrows():
            keys = self.find_prefix(group.prefix)
            # under the prefix filter
            filtering = mapper.named.isin(keys)
            # under the contains keyword filter
            if group.kw != "[ALL]":
                filtering *= mapper.named.apply(
                    lambda x: group.kw in x.split("."))
            mapper['group'].loc[filtering] = group.sn
        return dict(group_df=group_df, mapper=mapper)
        # create param_groups

    def create_param_groups(self, prefix_dict: Dict[str, Dict[str, Any]]):
        dfs = self.grouping(prefix_dict)
        group_df = dfs["group_df"]
        mapper = dfs["mapper"]

        param_groups = []
        # iter over groups
        for idx, group in group_df.iterrows():
            params = []
            if group.conf.get("freeze"):
                freeze = True
            else:
                freeze = False
            kw = group.kw
            conf = self.create_conf_dict(group.conf)
            sub_mapper = mapper[mapper.group == group.sn]
            if len(sub_mapper) == 0:
                continue
            for sub_idx, sub in sub_mapper.iterrows():
                parameter = self.param_dict[sub.named]
                if freeze:
                    parameter.requires_grad = False
                else:
                    parameter.requires_grad = True
                    params.append(parameter)
            if len(params) > 0:
                param_groups.append(dict(
                    params=params,
                    **conf
                ))
        default_mapper = mapper[mapper.group == "*"]
        # iter over the remaining
        if len(default_mapper) > 0:
            params = []
            for sub_idx, sub in default_mapper.iterrows():
                params.append(self.param_dict[sub.named])
            param_groups.append(dict(params=params))

        return param_groups

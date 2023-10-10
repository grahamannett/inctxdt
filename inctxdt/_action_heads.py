        # self.action_head_mods = nn.ModuleDict(
        #     {
        #         "norm": nn.LayerNorm(embedding_dim),
        #         "linear": nn.Linear(embedding_dim, 1),
        #         "activation": nn.Tanh(),  # all values for actions are scaled to -1 to -1
        #     }
        # )
        # self.norm_out = nn.LayerNorm(embedding_dim)
        # self._action_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Tanh())


        self.conv_head = nn.ModuleDict(
            {
                "norm": nn.LayerNorm(embedding_dim),
                "conv2d": nn.Conv2d(embedding_dim, action_dim, kernel_size=(1, 2)),
                "activation": nn.Tanh(),
                "linear": nn.Linear(embedding_dim, 1),
            }
        )



def action_head(
        self,
        x: torch.Tensor,
        act_dim: int = None,  # number of actions needed per pred step
        seq_len: int = None,
        batch_size: int = None,
        vals_in_timestep: int = 3,
    ) -> torch.Tensor:
        # act_dim = act_dim or self.action_dim
        # seq_len = seq_len or (x.shape[1] // 3)

        # possible heads:
        # return self._action_head2d(x, act_dim, seq_len, batch_size, vals_in_timestep)
        # return self._alt(x, batch_size, seq_len)

        # notes:  i cant tell if i should be using permute or view or what
        return self._alt_conv_(x, batch_size=batch_size, seq_len=seq_len)

        # 1d filter
        # x = x.reshape(batch_size, -1, self.embedding_dim)
        # x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), seq_len * act_dim).permute(0, 2, 1)
        # x_postnorm = self.head["norm"](x)
        # x_out = self.head["activation"](self.head["linear"](x_postnorm))
        # x_out = x_out.reshape(
        #     batch_size, seq_len, act_dim
        # )  # reshape because the results expect act dim to not be folded in
        # return x_out

    def _alt(self, x: torch.Tensor, batch_size: int, seq_length: int, **kwargs):
        # out = self.action_head(out[:, 1::3]) * self.max_action
        x = x.reshape(batch_size, seq_length, 3, self.embedding_dim).permute(0, 2, 1, 3)

        retrs = x[:, 0]
        states = x[:, 1]
        acts = x[:, 2]
        pred = retrs + states  # adding acts here makes it worse... trying to wonder why
        pred = self.head["norm"](pred)  # [here we have batch_size, seq_len, emb_dim]
        pred = self.head["activation"](self.head["linear_act"](pred))
        return pred

    def _alt_conv_(self, x: torch.Tensor, batch_size: int, seq_len: int, **kwargs):
        x = x.reshape(batch_size, seq_len, 3, self.embedding_dim).permute(0, 2, 1, 3)
        x_postnorm = self.conv_head["norm"](x)

        # take returns  and states and then put in [batch_size x emb_dim x seq_len x 2]
        x_postnorm = torch.stack([x_postnorm[:, 0], x_postnorm[:, 1]], dim=1)
        x_postnorm = x_postnorm.permute(0, 3, 2, 1)
        x_postnorm = self.conv_head["conv2d"](x_postnorm)
        x_postnorm = x_postnorm.squeeze(-1).permute(0, 2, 1).reshape(batch_size, -1)

        # I THINK THE ONE BELOW IS RIGHT
        # x_postnorm = x_postnorm.squeeze(-1).permute(0, 2, 1) # put back to [batch_size, seq_len, act_dim]

        return self.conv_head["activation"](x_postnorm)

    def _action_head1d(
        self, x: torch.Tensor, act_dim: int, seq_len: int, batch_size: int, vals_in_timestep: int = 3, **kwargs
    ) -> torch.Tensor:
        # this is slightly right - at least it works
        # x comes in as [batch_size, seq_len * 3, emb_dim]
        x = x.reshape(batch_size, -1, self.embedding_dim)
        x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), seq_len * act_dim).permute(0, 2, 1)
        x_postnorm = self.head["norm"](x)
        x_out = self.head["activation"](self.head["linear"](x_postnorm))
        x_out = x_out.reshape(
            batch_size, seq_len, act_dim
        )  # reshape because the results expect act dim to not be folded in
        return x_out

    def _action_head2d(
        self, x: torch.Tensor, act_dim: int, seq_len: int, batch_size: int, vals_in_timestep: int = 3, **kwargs
    ) -> torch.Tensor:
        # this is slightly right - at least it works
        # x comes in as [batch_size, seq_len * 3, emb_dim]
        x_postnorm = self.head["norm"](x)

        # convert to [batch_size, seq_len, 3, emb_dim] - view seems to work but is reshape better !!!
        x_postnorm = x_postnorm.view(
            batch_size, -1, vals_in_timestep, self.embedding_dim
        )  # note: using reshape() rather than view here makes it worse.  i think the reshape makes the seq fall out

        # [batch_size, 3, seq_len, emb_dim]
        x_out = F.adaptive_avg_pool2d(x_postnorm, (act_dim, self.embedding_dim))

        x_out = self.head["activation"](self.head["linear"](x_out))
        return x_out

class Meta(nn.Module):
    def __init__(self, args, config):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Classifier(config)
        self.net = self.net.to(device)

        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

        self.method = args.method

    def forward_ProtoMAML(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat):
        """
        b: number of tasks
        setsz: the size for each task

        :param x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_spt:   [b, setsz]
        :param x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = len(x_spt)
        querysz = len(y_qry[0])
        losses_s = [0 for _ in range(self.update_step)]
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            feat_spt = torch.Tensor(np.vstack(([feat[g_spt[i][j]][np.array(x)] for j, x in enumerate(n_spt[i])]))).to(
                device)
            feat_qry = torch.Tensor(np.vstack(([feat[g_qry[i][j]][np.array(x)] for j, x in enumerate(n_qry[i])]))).to(
                device)
            # 1. run the i-th task and compute loss for k=0
            logits, _ = self.net(x_spt[i].to(device), c_spt[i].to(device), feat_spt, vars=None)
            loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
            losses_s[0] += loss
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, self.net.parameters())
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + acc_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, fast_weights)
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, _ = self.net(x_spt[i].to(device), c_spt[i].to(device), feat_spt, fast_weights)
                loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
                losses_s[k] += loss
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        if torch.isnan(loss_q):
            pass
        else:
            # optimize theta parameters
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.meta_optim.step()

        accs = np.array(corrects) / (task_num)

        return accs

    def finetunning_ProtoMAML(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat):
        querysz = len(y_qry[0])

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        x_spt = x_spt[0]
        y_spt = y_spt[0]
        x_qry = x_qry[0]
        y_qry = y_qry[0]
        c_spt = c_spt[0]
        c_qry = c_qry[0]
        n_spt = n_spt[0]
        n_qry = n_qry[0]
        g_spt = g_spt[0]
        g_qry = g_qry[0]

        feat_spt = torch.Tensor(np.vstack(([feat[g_spt[j]][np.array(x)] for j, x in enumerate(n_spt)]))).to(device)
        feat_qry = torch.Tensor(np.vstack(([feat[g_qry[j]][np.array(x)] for j, x in enumerate(n_qry)]))).to(device)

        # 1. run the i-th task and compute loss for k=0
        logits, _ = net(x_spt.to(device), c_spt.to(device), feat_spt)
        loss, _, prototypes = proto_loss_spt(logits, y_spt, self.k_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q, _ = net(x_qry.to(device), c_qry.to(device), feat_qry, net.parameters())
            loss_q, acc_q = proto_loss_qry(logits_q, y_qry, prototypes)
            corrects[0] = corrects[0] + acc_q
        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q, _ = net(x_qry.to(device), c_qry.to(device), feat_qry, fast_weights)
            loss_q, acc_q = proto_loss_qry(logits_q, y_qry, prototypes)
            corrects[1] = corrects[1] + acc_q

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits, _ = net(x_spt.to(device), c_spt.to(device), feat_spt, fast_weights)
            loss, _, prototypes = proto_loss_spt(logits, y_spt, self.k_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q, _ = net(x_qry.to(device), c_qry.to(device), feat_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q, acc_q = proto_loss_qry(logits_q, y_qry, prototypes)
            corrects[k + 1] = corrects[k + 1] + acc_q

        del net
        accs = np.array(corrects)

        return accs

    def forward(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat):
        if self.method == 'G-Meta':
            accs = self.forward_ProtoMAML(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat):
        if self.method == 'G-Meta':
            accs = self.finetunning_ProtoMAML(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,
                                              feat)
        return accs

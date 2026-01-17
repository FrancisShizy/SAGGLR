if loss_type == "MSE": # only considering MSE of predicted vs. true activity scores
    if regularization:
        loss += lambda_MSE * lasso_regular_penalty(model, final_layer_params)
    else:
        pass
else:
    if loss_type == "MSE+UCN": # considering MSE & uncommon node information
        loss += lambda1 * loss_uncommon_node(data_i, data_j, model)
    elif loss_type == "MSE+N": # considering MSE & all node information
        if regularization:
            if Sparse: # sparse group lasso 
                loss += loss_node_with_sparse_group_lasso(data_i, data_j, model, lambda_group, common_group_params, uncommon_group_params, alpha)
                loss += alpha * lambda_group * lasso_regular_penalty(model, common_group_params)
                loss += alpha * lambda_group * lasso_regular_penalty(model, uncommon_group_params)
                # loss += lambda_group * lasso_regular_penalty(model, final_layer_params)
            else: # group lasso
                loss += loss_node_with_group_lasso(data_i, data_j, model, lambda_group, common_group_params, uncommon_group_params)
        else: # no group lasso
            loss += lambda1 * loss_common_node(data_i, data_j, model)
    elif loss_type == "MSE+AC":
            loss += MSE_LOSS_FN(
                out_i - out_j, data_i.y.to(DEVICE) - data_j.y.to(DEVICE)
            )
            if regularization:
                loss += lambda_MSE * lasso_regular_penalty(model, final_layer_params)
    elif loss_type == "MSE+UCNlocal":
            loss_batch, n_subs_in_batch = loss_node_local(
                data_i, data_j, model
            )
            loss += lambda1 * loss_batch
            # argue whether to use regularization
            if regularization:
                loss += lambda_MSE * lasso_regular_penalty(model, final_layer_params)

            loss.backward()
            loss_all += loss.item() * n_subs_in_batch

            optimizer.step()
            progress.set_postfix({"loss": loss.item()})
            total += n_subs_in_batch
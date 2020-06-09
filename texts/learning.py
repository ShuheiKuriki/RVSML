from rvsml.run_RVSML import run_RVSML
from rvsml.RVSML_OT_Learning import RVSML_OT_Learning
from src.utils import bool_flag,initialize_exp,read_txt_embeddings,loadFile
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator
from collections import OrderedDict
import logging,io,os,time,json,argparse,torch
import numpy as np

def learning(params,src_data,tgt_data,options):
    VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'
    logger = logging.getLogger('{}Log'.format(src_data.dataname))
    for i in range(10):
        # tic = time.time()
        if i==0:
            options.initialize = True
        else:
            options.initialize = False
        
        logger.info("src_learning {}回目".format(i+1))
        src_data = RVSML_OT_Learning(src_data,options,params)
        
        logger.info("tgt_learning {}回目".format(i+1))
        tgt_data = RVSML_OT_Learning(tgt_data,options,params)
        
        # build model / trainer / evaluator
        src_emb, tgt_emb, mapping, discriminator = build_model(params, src_data, tgt_data, True)
        
        trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)
        evaluator = Evaluator(trainer)


        """
        Learning loop for Adversarial Training
        """
        logger.info('----> ADVERSARIAL TRAINING <----\n\n')

        # training loop
        for n_epoch in range(params.n_epochs):

            logger.info('Starting adversarial training epoch %i...' % n_epoch)
            tic = time.time()
            n_words_proc = 0
            stats = {'DIS_COSTS': []}

            for n_iter in range(0, params.epoch_size, params.batch_size):

                # discriminator training
                for _ in range(params.dis_steps):
                    trainer.dis_step(stats)

                # mapping training (discriminator fooling)
                n_words_proc += trainer.mapping_step(stats)

                # log stats
                if n_iter % 500 == 0:
                    stats_str = [('DIS_COSTS', 'Discriminator loss')]
                    stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                                for k, v in stats_str if len(stats[k]) > 0]
                    stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                    logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))

                    # reset
                    tic = time.time()
                    n_words_proc = 0
                    for k, _ in stats_str:
                        del stats[k][:]

            # embeddings / discriminator evaluation
            to_log = OrderedDict({'n_epoch': n_epoch})
            evaluator.all_eval(to_log)
            evaluator.eval_dis(to_log)

            # JSON log / save best model / end of epoch
            logger.info("__log__:%s" % json.dumps(to_log))
            trainer.save_best(to_log, VALIDATION_METRIC)
            logger.info('End of epoch %i.\n\n' % n_epoch)

            # update the learning rate (stop if too small)
            trainer.update_lr(to_log, VALIDATION_METRIC)
            if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
                logger.info('Learning rate < 1e-6. BREAK.')
                break


        """
        Learning loop for Procrustes Iterative Refinement
        """
        # if params.n_refinement > 0:
        # Get the best mapping according to VALIDATION_METRIC
        logger.info('----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
        trainer.reload_best()

        # training loop
        for n_iter in range(params.n_refinement):

            logger.info('Starting refinement iteration %i...' % n_iter)

            # build a dictionary from aligned embeddings
            trainer.build_dictionary()

            # apply the Procrustes solution
            trainer.procrustes()

            # embeddings evaluation
            to_log = OrderedDict({'n_iter': n_iter})
            evaluator.all_eval(to_log)

            # JSON log / save best model / end of epoch
            logger.info("__log__:%s" % json.dumps(to_log))
            trainer.save_best(to_log, VALIDATION_METRIC)
            logger.info('End of refinement iteration %i.\n\n' % n_iter)


        src_data.trans_mat = torch.mm(src_data.trans_mat, trainer.mapping.weight.data)

    return src_data,tgt_data
    # export embeddings
    # if params.export:
    #     trainer.reload_best()
    #     trainer.export()

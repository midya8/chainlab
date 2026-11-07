"""Offline synthetic sequence fixture; no external model or performance claim."""
import math
import random
from .core import fit,infer,viterbi


def fixture():
    truth={'version':'chainlab_v1','states':['quiet','active'],'symbols':['low','mid','high'],
           'initial':[.65,.35],'transition':[[.93,.07],[.1,.9]],'emission':[[.8,.17,.03],[.03,.17,.8]]}
    initial={'version':'chainlab_v1','states':['state0','state1'],'symbols':list(truth['symbols']),
             'initial':[.5,.5],'transition':[[.65,.35],[.35,.65]],'emission':[[.55,.3,.15],[.15,.3,.55]]}
    rng=random.Random(23)
    def draw(probabilities):
        u=rng.random();cumulative=0.
        for i,p in enumerate(probabilities):
            cumulative+=p
            if u<cumulative:return i
        return len(probabilities)-1
    def sample(count,length):
        sequences=[]
        for _ in range(count):
            state=draw(truth['initial']);seq=[]
            for t in range(length):
                if t:state=draw(truth['transition'][state])
                seq.append(truth['symbols'][draw(truth['emission'][state])])
            sequences.append(seq)
        return sequences
    return truth,initial,sample(20,40),sample(12,40)


def run_demo():
    truth,initial,train,test=fixture();report=fit(initial,train,iterations=30,tolerance=1e-6)
    total=sum(map(len,test));arms={}
    for name,model in [('supplied_initialization',initial),('trained_hmm',report['model']),('generating_model_reference',truth)]:
        arms[name]={'test_mean_negative_log_likelihood':-math.fsum(infer(model,seq)['log_likelihood'] for seq in test)/total}
    counts={s:sum(seq.count(s) for seq in train) for s in initial['symbols']};n=sum(counts.values())
    # iid control uses training-only empirical symbol probabilities, no HMM fit.
    arms['training_iid_symbol_control']={'test_mean_negative_log_likelihood':
        -math.fsum(math.log(counts[s]/n) for seq in test for s in seq)/total}
    long=['low']*256;long_result=infer(report['model'],long)
    sample=test[0];posterior=infer(report['model'],sample);path=viterbi(report['model'],sample)
    counterexample={'version':'chainlab_v1','states':['A','B','C'],'symbols':['x'],
                    'initial':[.4,.35,.25],'transition':[[0,.5,.5],[1,0,0],[1,0,0]],'emission':[[1],[1],[1]]}
    invalid_modes=infer(counterexample,['x','x']);valid_path=viterbi(counterexample,['x','x'])
    rare={'version':'chainlab_v1','states':['only'],'symbols':['rare','common'],
          'initial':[1],'transition':[[1]],'emission':[[1e-6,1-1e-6]]}
    rare_ll=infer(rare,['rare']*256)['log_likelihood']
    return {'seed':23,'train_sequences':20,'test_sequences':12,'length_per_sequence':40,
            'arms':arms,'training':{'history':report['history'],'stop_reason':report['stop_reason'],
                                    'initial_log_likelihood':report['history'][0]['log_likelihood'],
                                    'final_log_likelihood':report['history'][-1]['log_likelihood'],
                                    'diagnostics':report['diagnostics'],'model':report['model']},
            'long_low_sequence':{'length':len(long),'log_likelihood':long_result['log_likelihood'],
                                 'posterior_rows':len(long_result['state_posteriors'])},
            'rare_symbol_sequence':{'length':256,'log_likelihood':rare_ll,'exp_log_likelihood':math.exp(rare_ll)},
            'posterior_mode_counterexample':{'model':counterexample,'symbols':['x','x'],
                                            'state_posteriors':invalid_modes['state_posteriors'],
                                            'posterior_modes':invalid_modes['posterior_modes'],
                                            'viterbi_path':valid_path['path'],
                                            'viterbi_path_posterior':math.exp(valid_path['log_path_posterior']),
                                            'reason':'Marginal modes A/A use a forbidden A-to-A transition; Viterbi B/A is valid.'},
            'test_sequence_example':{'symbols':sample,'posterior_modes':posterior['posterior_modes'],
                                     'viterbi_path':path['path'],'log_path_posterior':path['log_path_posterior']},
            'scope':'Constructed two-regime categorical HMM data, one seed and supplied asymmetric initialization; '
                    'no hyperparameter/initialization search. Training-only EM, test sequence likelihood report-only. '
                    'Generating model is an oracle reference, not a learned competitor or guaranteed best finite-sample score. '
                    'iid control learns empirical symbol frequencies with no state transitions. No universal ranking, '
                    'true-state recovery, production anomaly-detection or causal interpretation claim.'}


def run_online_demo():
    from .online import filter_sequence,forecast
    # Absorbing states make future evidence's effect on an earlier state exact.
    example={'version':'chainlab_v1','states':['A','B'],'symbols':['x','y'],
             'initial':[.5,.5],'transition':[[1,0],[0,1]],'emission':[[1,0],[.5,.5]]}
    sequence=['x','y'];filtered=filter_sequence(example,sequence);smoothed=infer(example,sequence)
    truth,initial,train,test=fixture();trained=fit(initial,train,iterations=30,tolerance=1e-6)['model']
    online_ll=math.fsum(filter_sequence(trained,seq)['log_likelihood'] for seq in test)
    batch_ll=math.fsum(infer(trained,seq)['log_likelihood'] for seq in test)
    return {'future_information_counterexample':{'model':example,'sequence':sequence,
               'filtered_state_posteriors':filtered['filtered_state_posteriors'],
               'full_sequence_smoothed_posteriors':smoothed['state_posteriors'],
               'prior_predictive_symbol_probabilities':filtered['prior_predictive_symbol_probabilities'],
               'observed_log_predictive':filtered['observed_log_predictive'],
               'forecast_after_x':forecast(example,['x'],3),
               'scope':'At time0, x-only filtering gives P(A)=2/3. Full x/y smoothing gives P(A)=0, using future y. '
                       'Using that smoothed row for an online decision at time0 would leak future evidence.'},
            'trained_fixture_likelihood_audit':{'test_sequences':len(test),'observations':sum(map(len,test)),
               'prefix_log_predictive_sum':online_ll,'forward_backward_log_likelihood_sum':batch_ll,
               'absolute_difference':abs(online_ll-batch_ll),
               'mean_negative_log_predictive':-online_ll/sum(map(len,test))},
            'trained_prefix_example':{'conditioning_symbols':test[0][:10],
                                      'forecast':forecast(trained,test[0][:10],5)},
            'scope':'Same seed23 development fixture and supplied training protocol as the main demo. '
                    'Prequential scores reset for each independent test sequence; no test fitting or future symbol conditioning. '
                    'Equality of likelihood computations checks implementation, not predictive quality or calibration.'}

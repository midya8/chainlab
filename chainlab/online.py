"""Prefix-only HMM filtering and unobserved-future marginal forecasts."""
import math
from .core import validate_model,_sequence,_log,_lse,_normalized


def _filter_encoded(model,obs):
    n=len(model['states']);m=len(model['symbols'])
    a=[[_log(v) for v in row] for row in model['transition']]
    b=[[_log(v) for v in row] for row in model['emission']]
    log_prior=[_log(v) for v in model['initial']]
    filtered=[];predictive=[];scores=[];prefix=[];total=0.;mass_error=0.;log_posterior=[]
    for t,symbol in enumerate(obs):
        if t:log_prior=[_lse([log_posterior[i]+a[i][j] for i in range(n)]) for j in range(n)]
        log_predictive=[_lse([log_prior[i]+b[i][s] for i in range(n)]) for s in range(m)]
        score=log_predictive[symbol]
        if score==-math.inf:raise ValueError('observation prefix has zero probability under this model; no smoothing fallback')
        log_posterior=[log_prior[i]+b[i][symbol]-score for i in range(n)]
        raw=[math.exp(v) for v in log_posterior];raw_predictive=[math.exp(v) for v in log_predictive]
        mass_error=max(mass_error,abs(math.fsum(raw)-1),abs(math.fsum(raw_predictive)-1))
        filtered.append(_normalized(raw));predictive.append(_normalized(raw_predictive))
        scores.append(score);total+=score;prefix.append(total)
    return {'filtered_state_posteriors':filtered,'prior_predictive_symbol_probabilities':predictive,
            'observed_log_predictive':scores,'prefix_log_likelihoods':prefix,'log_likelihood':total,
            'states':list(model['states']),'symbols':list(model['symbols']),
            'diagnostics':{'max_pre_normalization_mass_error':mass_error},
            'scope':'Each filtered row uses observations only through its own time; each predictive row uses only the preceding prefix. '
                    'Log posterior messages stay in log space between steps. Probability outputs are numerically normalized and can '
                    'underflow tiny entries to zero; observed log scores remain the authoritative likelihood representation. '
                    'This batch prefix audit is not a persistent streaming session, smoothed inference or a model-quality guarantee.'},log_posterior


def filter_sequence(model,sequence):
    model=validate_model(model);obs=_sequence(sequence,model['symbols'])
    return _filter_encoded(model,obs)[0]


def forecast(model,sequence,horizon=1):
    model=validate_model(model);obs=_sequence(sequence,model['symbols'])
    if type(horizon) is not int or not 1<=horizon<=32:raise ValueError('horizon must be integer1-32')
    filtered,log_state=_filter_encoded(model,obs);n=len(model['states']);m=len(model['symbols'])
    a=[[_log(v) for v in row] for row in model['transition']]
    b=[[_log(v) for v in row] for row in model['emission']];steps=[]
    for h in range(1,horizon+1):
        log_state=[_lse([log_state[i]+a[i][j] for i in range(n)]) for j in range(n)]
        normalizer=_lse(log_state);log_state=[v-normalizer for v in log_state]
        log_symbols=[_lse([log_state[i]+b[i][s] for i in range(n)]) for s in range(m)]
        steps.append({'horizon':h,'state_probabilities':_normalized([math.exp(v) for v in log_state]),
                      'symbol_probabilities':_normalized([math.exp(v) for v in log_symbols])})
    return {'states':list(model['states']),'symbols':list(model['symbols']),
            'conditioning_observations':len(obs),'conditioning_log_likelihood':filtered['log_likelihood'],
            'last_filtered_state_posterior':filtered['filtered_state_posteriors'][-1],'forecasts':steps,
            'scope':'Horizon1 transitions once from the last conditioned state, then predicts its emission. '
                    'Each horizon is a marginal conditioned on the observed prefix, not a joint future path or independent forecast. '
                    'No guessed future observation is fed back as evidence. Numerically normalized probability outputs may '
                    'underflow tiny entries. Fixed HMM assumptions, not a calibration, deployment or ground-truth guarantee.'}

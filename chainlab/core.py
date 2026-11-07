"""Discrete HMM inference and unsmoothed multi-sequence maximum-likelihood EM."""
import math


def _names(value,maximum,label):
    if not isinstance(value,list) or not 1<=len(value)<=maximum or any(type(x) is not str or not x or len(x)>64 for x in value) or len(set(value))!=len(value):
        raise ValueError(label+' must be unique nonempty names, at most '+str(maximum))
    return list(value)


def _probabilities(row,width):
    if not isinstance(row,list) or len(row)!=width or any(type(x) not in (int,float) or not 0<=x<=1 or not math.isfinite(x) for x in row):
        raise ValueError('probability vector has invalid shape or values')
    if abs(math.fsum(row)-1)>1e-12:raise ValueError('probability vectors must sum to1 within1e-12; no silent renormalization')
    return [float(x) for x in row]


def validate_model(model):
    keys={'version','states','symbols','initial','transition','emission'}
    if not isinstance(model,dict) or set(model)!=keys or model['version']!='chainlab_v1':raise ValueError('invalid chainlab_v1 model schema')
    states=_names(model['states'],8,'states');symbols=_names(model['symbols'],16,'symbols');n=len(states)
    result={'version':'chainlab_v1','states':states,'symbols':symbols,'initial':_probabilities(model['initial'],n)}
    for key,width in [('transition',n),('emission',len(symbols))]:
        if not isinstance(model[key],list) or len(model[key])!=n:raise ValueError(key+' must have one row per state')
        result[key]=[_probabilities(row,width) for row in model[key]]
    return result


def _sequence(value,symbols):
    if not isinstance(value,list) or not 1<=len(value)<=256:raise ValueError('sequence must contain1-256 symbol names')
    lookup={s:i for i,s in enumerate(symbols)}
    if any(type(s) is not str or s not in lookup for s in value):raise ValueError('sequence has unknown/non-string symbol')
    return [lookup[s] for s in value]


def _log(p):return math.log(p) if p else -math.inf


def _lse(values):
    peak=max(values)
    if peak==-math.inf:return peak
    return peak+math.log(math.fsum(math.exp(v-peak) for v in values))


def _normalized(values):
    total=math.fsum(values)
    if total<=0 or not math.isfinite(total):raise ValueError('cannot normalize zero/nonfinite posterior mass')
    return [v/total for v in values]


def _infer(model,obs):
    n=len(model['states']);length=len(obs)
    a=[[_log(p) for p in row] for row in model['transition']]
    b=[[_log(p) for p in row] for row in model['emission']]
    forward=[[_log(model['initial'][i])+b[i][obs[0]] for i in range(n)]]
    for symbol in obs[1:]:
        forward.append([b[j][symbol]+_lse([forward[-1][i]+a[i][j] for i in range(n)]) for j in range(n)])
    likelihood=_lse(forward[-1])
    if likelihood==-math.inf:raise ValueError('observation sequence has zero probability under this model; no smoothing fallback')
    backward=[[0.]*n for _ in range(length)]
    for t in reversed(range(length-1)):
        backward[t]=[_lse([a[i][j]+b[j][obs[t+1]]+backward[t+1][j] for j in range(n)]) for i in range(n)]
    gamma=[];xi=[];mass_error=0.
    for t in range(length):
        raw=[math.exp(forward[t][i]+backward[t][i]-likelihood) for i in range(n)]
        mass_error=max(mass_error,abs(math.fsum(raw)-1));gamma.append(_normalized(raw))
    for t in range(length-1):
        raw=[math.exp(forward[t][i]+a[i][j]+b[j][obs[t+1]]+backward[t+1][j]-likelihood) for i in range(n) for j in range(n)]
        mass_error=max(mass_error,abs(math.fsum(raw)-1));normalized=_normalized(raw)
        xi.append([normalized[i*n:(i+1)*n] for i in range(n)])
    return likelihood,gamma,xi,mass_error


def infer(model,sequence):
    model=validate_model(model);obs=_sequence(sequence,model['symbols'])
    likelihood,gamma,xi,error=_infer(model,obs)
    modes=[max(range(len(row)),key=row.__getitem__) for row in gamma]
    return {'log_likelihood':likelihood,'state_posteriors':gamma,'transition_posteriors':xi,
            'posterior_modes':[model['states'][i] for i in modes],
            'diagnostics':{'max_pre_normalization_mass_error':error},
            'scope':'Full-sequence smoothed state/adjacent-state marginals, not online filtering. '
                    'Posterior modes maximize per-position marginals, not path probability; may violate zero-transition constraints. '
                    'Posterior rows are numerically renormalized; input model probabilities are not.'}


def viterbi(model,sequence):
    model=validate_model(model);obs=_sequence(sequence,model['symbols']);n=len(model['states'])
    a=[[_log(p) for p in row] for row in model['transition']];b=[[_log(p) for p in row] for row in model['emission']]
    score=[_log(model['initial'][i])+b[i][obs[0]] for i in range(n)];pointers=[]
    for symbol in obs[1:]:
        new=[];back=[]
        for j in range(n):
            candidates=[score[i]+a[i][j] for i in range(n)]
            i=max(range(n),key=candidates.__getitem__);new.append(candidates[i]+b[j][symbol]);back.append(i)
        score=new;pointers.append(back)
    final=max(range(n),key=score.__getitem__);joint=score[final]
    if joint==-math.inf:raise ValueError('observation sequence has zero probability under this model; no path')
    path=[final]
    for back in reversed(pointers):path.append(back[path[-1]])
    path.reverse();likelihood,_,_,_=_infer(model,obs)
    return {'path':[model['states'][i] for i in path],'state_indices':path,'log_joint_probability':joint,
            'log_observation_likelihood':likelihood,'log_path_posterior':joint-likelihood,
            'scope':'Single maximum-joint-probability path, equivalently maximum posterior path for fixed observations. '
                    'Exact local predecessor/final ties keep the first state index; not a promised lexicographic whole-path tie rule. '
                    'Not per-position posterior modes or a guarantee of ground-truth state recovery.'}


def fit(model,sequences,iterations=25,tolerance=1e-6):
    model=validate_model(model)
    if not isinstance(sequences,list) or not 1<=len(sequences)<=32:raise ValueError('training requires1-32 independent nonempty sequences')
    encoded=[_sequence(s,model['symbols']) for s in sequences];total=sum(map(len,encoded))
    if total>2048:raise ValueError('training allows at most2048 observations total')
    if type(iterations) is not int or not 1<=iterations<=100:raise ValueError('iterations must be integer1-100')
    if type(tolerance) not in (int,float) or not 0<=tolerance<=1 or not math.isfinite(tolerance):raise ValueError('tolerance must be finite in[0,1]')
    n=len(model['states']);m=len(model['symbols'])
    likelihood=math.fsum(_infer(model,obs)[0] for obs in encoded)
    history=[{'iteration':0,'log_likelihood':likelihood}];stop='iteration_budget';rejected=None
    for step in range(1,iterations+1):
        start=[0.]*n;trans=[[0.]*n for _ in range(n)];emit=[[0.]*m for _ in range(n)]
        for obs in encoded:
            _,gamma,xi,_=_infer(model,obs)
            for i in range(n):
                start[i]+=gamma[0][i]
                for t,symbol in enumerate(obs):emit[i][symbol]+=gamma[t][i]
                for j in range(n):trans[i][j]+=math.fsum(pair[i][j] for pair in xi)
        candidate=dict(model,initial=_normalized(start));retained_transition=[];retained_emission=[]
        candidate['transition']=[];candidate['emission']=[]
        for i in range(n):
            if math.fsum(trans[i])==0:
                candidate['transition'].append(list(model['transition'][i]));retained_transition.append(i)
            else:candidate['transition'].append(_normalized(trans[i]))
            if math.fsum(emit[i])==0:
                candidate['emission'].append(list(model['emission'][i]));retained_emission.append(i)
            else:candidate['emission'].append(_normalized(emit[i]))
        validate_model(candidate)
        new_likelihood=math.fsum(_infer(candidate,obs)[0] for obs in encoded);improvement=new_likelihood-likelihood
        if improvement<0:
            if improvement < -1e-7:raise ValueError('EM proposal decreased log likelihood beyond numerical tolerance')
            rejected={'iteration':step,'log_likelihood':new_likelihood,'change':improvement}
            stop='tiny_numerical_decrease_rejected';break
        model=candidate;likelihood=new_likelihood
        history.append({'iteration':step,'log_likelihood':likelihood,'improvement':improvement,
                        'zero_occupancy_transition_rows_retained':retained_transition,
                        'zero_occupancy_emission_rows_retained':retained_emission})
        if improvement<=tolerance:stop='improvement_tolerance';break
    return {'model':model,'history':history,'stop_reason':stop,'rejected_proposal':rejected,
            'diagnostics':{'sequences':len(encoded),'observations':total,'within_sequence_transitions':total-len(encoded),
                           'accepted_updates':len(history)-1},
            'scope':'Unsmoothed maximum-likelihood Baum-Welch with supplied initialization, no priors or validation selection. '
                    'Sequence starts reset initial-state counts; no transitions cross sequence boundaries. '
                    'Zero expected-occupancy rows retain their previous probabilities, explicitly reported. '
                    'Zero support cannot be resurrected by EM. Absent training symbols can lose emission support. '
                    'Accepted likelihood is nondecreasing; finite tolerance/budget/local optima and state-label symmetry remain.'}

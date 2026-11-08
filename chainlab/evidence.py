"""One explicit observation window, one prefix boundary, distinct inference objectives."""
from .core import infer,viterbi,validate_model,_sequence
from .online import filter_sequence,forecast

def inspect(model,sequence,cut,horizon=5):
    model=validate_model(model);_sequence(sequence,model['symbols'])
    if type(cut) is not int or not 1<=cut<=len(sequence):raise ValueError('cut must be integer1 through sequence length')
    online=filter_sequence(model,sequence)
    return dict(version=1,model=model,sequence=list(sequence),cut=cut,horizon=horizon,
                filtering=online,smoothing=infer(model,sequence),viterbi=viterbi(model,sequence),
                prefix_forecast=forecast(model,sequence[:cut],horizon),
                scope='Fixed supplied HMM. Filtering uses each own prefix; smoothing/Viterbi use the complete displayed window. '
                      'Forecast uses only the first cut observations; no guessed future evidence. '
                      'The displayed window may be a prefix of an original test sequence, not the whole original sequence. '
                      'State labels need not identify true states. No causal, calibration, new training or performance claim.')

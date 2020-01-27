using KnetLayers: AbstractRNN
using Knet: rnnforw
import KnetLayers: _forw

function _forw(rnn::AbstractRNN,seq::Array{T},h...;batchSizes=nothing,o...) where T<:Integer
    rnn.embedding === nothing && error("rnn has no embedding!")
    ndims(seq) == 1 && batchSizes === nothing && (seq = reshape(seq,1,length(seq)))
    y,hidden,memory,_ = rnnforw(rnn.specs,rnn.params,rnn.embedding(seq),h...;batchSizes=batchSizes,o...)
    return y,hidden,memory,nothing
end

function _forw(rnn::AbstractRNN,x,h...;o...)
    if rnn.embedding === nothing
        y,hidden,memory,_ = rnnforw(rnn.specs,rnn.params,x,h...;o...)
    else
        y,hidden,memory,_ = rnnforw(rnn.specs,rnn.params,rnn.embedding(x),h...;o...)
    end
    y,hidden,memory,nothing
end

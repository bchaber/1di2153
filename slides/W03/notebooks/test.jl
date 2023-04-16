struct Variable{T}
  v::T
end

import Base: convert, promote_rule
convert(::Type{Variable{T}}, x::Number) where T   = Variable(convert(T, x))
convert(::Type{Variable{T}}, x::Variable) where T = Variable(convert(T, x.v))
promote_rule(::Type{Variable{T}}, ::Type{R}) where {T,R} = Variable{promote_type(R,T)}

@show Variable{Float64}[1., 2, Variable{Integer}(0)]


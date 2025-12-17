# CorrectMatch.jl
# Copyright © 2019 Université catholique de Louvain, UCLouvain
# Copyright © 2019 Imperial College London
# by Luc Rocher, Julien Hendrickx, Yves-Alexandre de Montjoye
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import Base: show

abstract type CorrectMatchError <: Exception end

struct DataValidationError <: CorrectMatchError
    msg::String
end
show(io::IO, e::DataValidationError) = print(io, "DataValidationError: ", e.msg)

struct ModelFittingError <: CorrectMatchError
    msg::String
    cause::Union{Exception,Nothing}

    ModelFittingError(msg::String) = new(msg, nothing)
    ModelFittingError(msg::String, cause::Exception) = new(msg, cause)
end
show(io::IO, e::ModelFittingError) = begin
    print(io, "ModelFittingError: ", e.msg)
    if e.cause !== nothing
        print(io, " (caused by: ", e.cause, ")")
    end
end

struct InsufficientDataError <: CorrectMatchError
    msg::String
end
show(io::IO, e::InsufficientDataError) = print(io, "InsufficientDataError: ", e.msg)

struct NumericalInstabilityError <: CorrectMatchError
    msg::String
    operation::String
end
show(io::IO, e::NumericalInstabilityError) = print(io, "NumericalInstabilityError in ", e.operation, ": ", e.msg)

function wrap_fitting_error(e::Exception, msg::String)
    e isa CorrectMatchError && rethrow(e)
    throw(ModelFittingError(msg, e))
end

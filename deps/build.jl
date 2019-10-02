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

import Compat.Sys: isunix, iswindows

using BinDeps
@BinDeps.setup

BinDeps.lower(s::Base.Process, c::BinDeps.SynchronousStepCollection) = nothing 

# set to true to support intel fortran compiler
useIntelFortran = false

# construct absolute path
depsdir  = splitdir(Base.source_path())[1]
builddir = joinpath(depsdir,"builds")
srcdir   = joinpath(depsdir,"src")


println("=== Building MVNDST ===")
println("depsdir  = $depsdir")
println("builddir = $builddir")
println("srcdir   = $srcdir")
println("useIntel = $useIntelFortran")


if !isdir(builddir)
	println("creating build directory")
	mkdir(builddir)
	if !isdir(builddir)
		error("Could not create build directory")
	end
end

src = joinpath(depsdir, "mvndst.f")
outfile = joinpath(builddir, "mvndst")

@static if isunix()
	@build_steps begin
		run(`curl -L -o mvndst.f https://raw.githubusercontent.com/scipy/scipy/master/scipy/stats/mvndst.f`)

		if useIntelFortran
			run(`ifort -O3 -xHost -fPIC -fpp -openmp -integer-size 64 -shared  $src -o $outfile`)
		else
			println("fortran version")
			run(`gfortran --version`)
			run(`gfortran -O3 -fPIC -cpp -fopenmp -fdefault-integer-8 -shared  $src -o $outfile`)
		end
	end
end


@static if iswindows()
	@build_steps begin
		download("https://raw.githubusercontent.com/scipy/scipy/master/scipy/stats/mvndst.f",
					   joinpath(depsdir, "mvndst.f"))

		println("fortran version")
		run(`gfortran --version`)
		run(`gfortran -O3 -cpp -fopenmp -fdefault-integer-8 -shared -DBUILD_DLL $src -o $outfile`)
	end
end

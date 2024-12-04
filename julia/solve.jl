function section_properties(stringer_locs)
    vec_len = size(stringer_locs, 1) + 4
    A = zeros(vec_len)
    y = zeros(vec_len)
    z = zeros(vec_len)
    Iyy = zeros(vec_len)
    Izz = zeros(vec_len)
    Iyz = zeros(vec_len)
    E = zeros(vec_len)

    ## rear spar
    A[1] = 0.09375
    y[1] = 3.9375
    z[1] = 0.875
    Iyy[1] = 0.00439453125
    Izz[1] = 0.0001220703125
    Iyz[1] = 0
    E[1] = 1926e3

    ## top skin
    A[2] = 0.12525
    y[2] = 2
    z[2] = 1.354
    Iyy[2] = 0.0007491
    Izz[2] = 0.16706
    Iyz[2] = 0
    E[2] = 2036e3

    ## front spar
    A[3] = 0.15625
    y[3] = 0.0625
    z[3] = 0.625
    Iyy[3] = 0.02034505208
    Izz[3] = 0.0002034505208
    Iyz[3] = 0
    E[3] = 1926e3

    ## bottom skin
    A[4] = 0.12597
    y[4] = 2
    z[4] = 0.23438
    Iyy[4] = 0.00263426
    Izz[4] = 0.16796
    Iyz[4] = 0.0209929
    E[4] = 2036e3

    for (i, str_pos) in pairs(stringer_locs)
        if str_pos < 4
            y[4+i] = 15 * cos(0.066865794705 * str_pos + 1.43706473738) + 2
            z[4+i] = 15 * sin(0.066865794705 * str_pos + 1.43706473738) - 13.6160687473
        else
            y[4+i] = 8 - str_pos
            z[4+i] = y[4+i] / 8
        end
    end

    y_bar = esum(E .* A .* y) / sum(E .* A)
    z_bar = sum(E .* A .* z) / sum(E .* A)
    return (
        y_bar=y_bar,
        z_bar=z_bar,
        Iyy=sum(E .* (Iyy .+ A .* (z .- z_bar) .^ 2)),
        Izz=sum(E .* (Izz .+ A .* (y .- y_bar) .^ 2)),
        Iyz=sum(E .* (Iyz .+ A .* (y .- y_bar) .* (z .- z_bar))),
    )
end


function deflection_y(mod_elastic, i_yy, i_zz, i_yz, load)
    return (i_yy / (mod_elastic .* (i_yy .* i_zz .- i_yz .^ 2))) .* (load .- 5) .* -32180.8333333
end

function deflection_z(mod_elastic, i_yy, i_zz, i_yz, load)
    return (i_zz / (mod_elastic .* (i_yy .* i_zz .- i_yz .^ 2))) .* (load .- 5) .* -32180.8333333
end

function shear_flow(load, y_bar)
    return (load .* (y_bar .- 1) .+ 5 .* y_bar - 10) ./ 4.136
end

function torsion(x, load, y_bar)
    return 0.00018401233 .* x .* shear_flow(load, y_bar)
end

function stress_shear(load, y_bar, thickness)
    return (load .* (y_bar .- 1) .+ 5 .* y_bar .- 10) ./ (4.136 .* thickness)
end

function stress_normal(x, y, z, load, i_yy, i_zz, i_yz)
    return .-y .* ((load .- 5) .* (x .- 45.75) .* i_yy ./ (i_yy .* i_zz .- i_yz .^ 2)) .+ z .* (
        (load .- 5) .* (x .- 45.75) .* i_zz ./ (i_yy .* i_zz .- i_yz .^ 2)
    )
end

function failed()
    shear = stress_shear(load, y_bar, thickness)
    normal = stress_normal(x, y, z, load, i_yy, i_zz, i_yz)

    f_1 = -0.038
    f_11 = 0.0096
    f_66 = 0.44

    if thickness < 1 / 8
        f_1 = -0.054
        f_11 = 0.0072
    end

    if thickness < 1 / 16
        f_1 = -0.048
        f_11 = 0.0078
        f_66 = 0.31
    end

    return f_1 .* normal .+ f_11 .* normal .^ 2 .+ f_66 .* shear .^ 2 >= 1
end

function count_stacked(stringer_locs)
    sum([count(==(loc), stringer_locs) for loc in unique(stringer_locs)]) - size(stringer_locs, 1)
end

function count_adjacent(stringer_locs)
    uniq = unique(stringer_locs)
    dup = circshift(uniq, 1)
    return sum(uniq - dup .<= 1 / 8)
end

function final_score(n_stringer, n_ribs, n_stacked, n_adj, load, def_q, def_max, twist)
    weight = 0.4369 .+ n_ribs .* 0.00213 .+ n_stringer .* 0.0146
    design = 100 .* (
        0.6 .* 8 ./ (n_stringer .+ 1)
        .+
        0.3 .* 15 ./ (n_ribs .+ 1)
        .-
        (n_stacked ./ 8 .+ n_adj ./ 8)
    )
    perf = (
        0.5 .* load ./ weight
        .+
        0.5 ./ def_q
        .+
        0.05 .* (load ./ def_max .+ load ./ twist)
        .-
        10 .* weight
    )
    return design .+ perf
end
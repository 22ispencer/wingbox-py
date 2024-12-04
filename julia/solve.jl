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

    y_bar = sum(E .* A .* y) / sum(E .* A)
    z_bar = sum(E .* A .* z) / sum(E .* A)
    return (
        y_bar=y_bar,
        z_bar=z_bar,
        Iyy=sum(E .* (Iyy .+ A .* (z .- z_bar) .^ 2)),
        Izz=sum(E .* (Izz .+ A .* (y .- y_bar) .^ 2)),
        Iyz=sum(E .* (Iyz .+ A .* (y .- y_bar) .* (z .- z_bar))),
    )
end
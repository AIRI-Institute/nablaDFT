import numpy as np


class LineSearch:
    def __init__(self, xtol=1e-14):
        self.xtol = xtol
        self.task = "START"
        self.fc = 0
        self.gc = 0
        self.case = 0
        self.old_stp = 0

    def _line_search(
        self,
        func,
        myfprime,
        xk,
        pk,
        gfk,
        old_fval,
        old_old_fval,
        n_configs,
        ats_mask,
        batch,
        n_ats_per_config,
        n_ats,
        configs_mask,
        maxstep=0.2,
        c1=0.23,
        c2=0.46,
        xtrapl=1.1,
        xtrapu=4.0,
        stpmax=50.0,
        stpmin=1e-8,
        max_abs_step=100,
        args=(),
    ):
        self.stpmin = stpmin
        self.pk = pk
        self.stpmax = stpmax
        self.xtrapl = xtrapl
        self.xtrapu = xtrapu
        self.maxstep = maxstep
        self.phi0 = old_fval

        # alpha1 = pymin(maxstep,1.01*2*(phi0-old_old_fval)/derphi0)
        alpha1 = np.ones(n_configs)
        self.no_update = configs_mask

        if isinstance(myfprime, type(())):
            # eps = myfprime[1]
            fprime = myfprime[0]
            # ??? newargs = (f,eps) + args
            gradient = False
        else:
            fprime = myfprime
            newargs = args
            gradient = True

        fval = old_fval
        gval = gfk

        steps = np.ones(n_configs)
        old_steps = np.zeros(n_configs)
        self.tasks = ["START"] * n_configs

        self.isave = np.zeros((n_configs, 2), np.intc)
        self.dsave = np.zeros((n_configs, 13), float)
        for at_idx in range(n_configs):
            p_at = self.pk[ats_mask[at_idx]].ravel()
            p_size = np.sqrt((p_at**2).sum())
            if p_size <= np.sqrt(n_ats_per_config[at_idx] * 1e-10):
                p_at /= p_size / np.sqrt(n_ats * 1e-10)
            self.pk[ats_mask[at_idx]] = p_at.reshape(-1, 3)
        abs_step = 0
        while True:
            abs_step += 1
            if abs_step > max_abs_step:
                break
            for at_idx in range(n_configs):
                if self.no_update[at_idx]:
                    continue

                phi0 = self.phi0[at_idx]
                pk = self.pk[ats_mask[at_idx]].ravel()
                derphi0 = np.dot(gval[ats_mask[at_idx]].ravel(), pk)
                self.dim = len(pk)
                self.gms = np.sqrt(self.dim) * maxstep
                stp = self.step(
                    steps[at_idx],
                    phi0,
                    derphi0,
                    c1,
                    c2,
                    pk,
                    old_steps[at_idx],
                    at_idx,
                    self.xtol,
                    self.isave,
                    self.dsave,
                )

                old_steps[at_idx] = steps[at_idx]
                if self.tasks[at_idx] == 'FG':
                    steps[at_idx] = stp
                else:
                    self.no_update[at_idx] = True
                    steps[at_idx] = self.determine_step_(pk)

            if not np.any([x[:2] == "FG" for x in self.tasks]):
                break
            alpha1 = np.repeat(steps, n_ats_per_config).reshape(self.pk.shape[0], -1)
            fval = func(xk + self.pk * alpha1, *args)
            self.fc += 1
            gval = fprime(xk + self.pk * alpha1, *newargs)
            if gradient:
                self.gc += 1
            else:
                self.fc += len(xk) + 1
            self.phi0 = fval
            self.old_stp = alpha1

        alpha1 = np.repeat(steps, n_ats_per_config)
        return alpha1, fval, old_fval, self.no_update

    def step(self, stp, f, g, c1, c2, pk, old_stp, at_idx, xtol, isave, dsave):
        if self.tasks[at_idx][:5] == "START":
            # Check the input arguments for errors.
            if stp < self.stpmin:
                self.tasks[at_idx] = "ERROR: STP .LT. minstep"
            if stp > self.stpmax:
                self.tasks[at_idx] = "ERROR: STP .GT. maxstep"
            if g >= 0:
                self.tasks[at_idx] = "ERROR: INITIAL G >= 0"
            if c1 < 0:
                self.tasks[at_idx] = "ERROR: c1 .LT. 0"
            if c2 < 0:
                self.tasks[at_idx] = "ERROR: c2 .LT. 0"
            if xtol < 0:
                self.tasks[at_idx] = "ERROR: XTOL .LT. 0"
            if self.stpmin < 0:
                self.tasks[at_idx] = "ERROR: minstep .LT. 0"
            if self.stpmax < self.stpmin:
                self.tasks[at_idx] = "ERROR: maxstep .LT. minstep"
            if self.tasks[at_idx][:5] == "ERROR":
                return stp

            # Initialize local variables.
            self.bracket = False
            stage = 1
            finit = f
            ginit = g
            gtest = c1 * ginit
            width = self.stpmax - self.stpmin
            width1 = width / 0.5
            #           The variables stx, fx, gx contain the values of the step,
            #           function, and derivative at the best step.
            #           The variables sty, fy, gy contain the values of the step,
            #           function, and derivative at sty.
            #           The variables stp, f, g contain the values of the step,
            #           function, and derivative at stp.
            stx = 0
            fx = finit
            gx = ginit
            sty = 0
            fy = finit
            gy = ginit
            stmin = 0
            stmax = stp + self.xtrapu * stp
            self.tasks[at_idx] = "FG"
            self.save(
                at_idx,
                (
                    stage,
                    ginit,
                    gtest,
                    gx,
                    gy,
                    finit,
                    fx,
                    fy,
                    stx,
                    sty,
                    stmin,
                    stmax,
                    width,
                    width1,
                ),
            )
            stp = self.determine_step(stp, old_stp, pk)
            # return stp, f, g
            return stp
        else:
            if self.isave[at_idx][0] == 1:
                self.bracket = True
            else:
                self.bracket = False
            stage = self.isave[at_idx][1]
            (
                ginit,
                gtest,
                gx,
                gy,
                finit,
                fx,
                fy,
                stx,
                sty,
                stmin,
                stmax,
                width,
                width1,
            ) = self.dsave[at_idx]

            #           If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
            #           algorithm enters the second stage.
            ftest = finit + stp * gtest
            if stage == 1 and f < ftest and g >= 0.0:
                stage = 2

            #           Test for warnings.
            if self.bracket and (stp <= stmin or stp >= stmax):
                self.tasks[at_idx] = "WARNING: ROUNDING ERRORS PREVENT PROGRESS"
            if self.bracket and stmax - stmin <= self.xtol * stmax:
                self.tasks[at_idx] = "WARNING: XTOL TEST SATISFIED"
            if stp == self.stpmax and f <= ftest and g <= gtest:
                self.tasks[at_idx] = "WARNING: STP = maxstep"
            if stp == self.stpmin and (f > ftest or g >= gtest):
                self.tasks[at_idx] = "WARNING: STP = minstep"

            #           Test for convergence.
            # print (f, ftest, abs(g), c2 * (- ginit))
            if f <= ftest and abs(g) <= c2 * (-ginit):
                self.tasks[at_idx] = "CONVERGENCE"

            #           Test for termination.
            if self.tasks[at_idx][:4] == "WARN" or self.tasks[at_idx][:4] == "CONV":
                self.save(
                    at_idx,
                    (
                        stage,
                        ginit,
                        gtest,
                        gx,
                        gy,
                        finit,
                        fx,
                        fy,
                        stx,
                        sty,
                        stmin,
                        stmax,
                        width,
                        width1,
                    ),
                )
                # return stp, f, g
                return stp

            #              A modified function is used to predict the step during the
            #              first stage if a lower function value has been obtained but
            #              the decrease is not sufficient.
            # if stage == 1 and f <= fx and f > ftest:
            #           #    Define the modified function and derivative values.
            #    fm =f - stp * gtest
            #    fxm = fx - stx * gtest
            #    fym = fy - sty * gtest
            #    gm = g - gtest
            #    gxm = gx - gtest
            #    gym = gy - gtest

            #               Call step to update stx, sty, and to compute the new step.
            #    stx, sty, stp, gxm, fxm, gym, fym = self.update (stx, fxm, gxm, sty,
            #                                        fym, gym, stp, fm, gm,
            #                                        stmin, stmax)

            #           #    Reset the function and derivative values for f.

            #    fx = fxm + stx * gtest
            #    fy = fym + sty * gtest
            #    gx = gxm + gtest
            #    gy = gym + gtest

            # else:
            #           Call step to update stx, sty, and to compute the new step.

            stx, sty, stp, gx, fx, gy, fy = self.update(stx, fx, gx, sty, fy, gy, stp, f, g, stmin, stmax, old_stp, pk)

            #           Decide if a bisection step is needed.

            if self.bracket:
                if abs(sty - stx) >= 0.66 * width1:
                    stp = stx + 0.5 * (sty - stx)
                width1 = width
                width = abs(sty - stx)

            #           Set the minimum and maximum steps allowed for stp.

            if self.bracket:
                stmin = min(stx, sty)
                stmax = max(stx, sty)
            else:
                stmin = stp + self.xtrapl * (stp - stx)
                stmax = stp + self.xtrapu * (stp - stx)

            #           Force the step to be within the bounds maxstep and minstep.

            stp = max(stp, self.stpmin)
            stp = min(stp, self.stpmax)

            if stx == stp and stp == self.stpmax and stmin > self.stpmax:
                self.no_update = True
            #           If further progress is not possible, let stp be the best
            #           point obtained during the search.

            if (self.bracket and stp < stmin or stp >= stmax) or (self.bracket and stmax - stmin < self.xtol * stmax):
                stp = stx

            #           Obtain another function and derivative.

            self.tasks[at_idx] = "FG"
            self.save(
                at_idx,
                (
                    stage,
                    ginit,
                    gtest,
                    gx,
                    gy,
                    finit,
                    fx,
                    fy,
                    stx,
                    sty,
                    stmin,
                    stmax,
                    width,
                    width1,
                ),
            )
            return stp

    def update(self, stx, fx, gx, sty, fy, gy, stp, fp, gp, stpmin, stpmax, old_stp, pk):
        sign = gp * (gx / abs(gx))

        #       First case: A higher function value. The minimum is bracketed.
        #       If the cubic step is closer to stx than the quadratic step, the
        #       cubic step is taken, otherwise the average of the cubic and
        #       quadratic steps is taken.
        if fp > fx:  # case1
            self.case = 1
            theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))
            gamma = s * np.sqrt((theta / s) ** 2.0 - (gx / s) * (gp / s))
            if stp < stx:
                gamma = -gamma
            p = (gamma - gx) + theta
            q = ((gamma - gx) + gamma) + gp
            r = p / q
            stpc = stx + r * (stp - stx)
            stpq = stx + ((gx / ((fx - fp) / (stp - stx) + gx)) / 2.0) * (stp - stx)
            if abs(stpc - stx) < abs(stpq - stx):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc) / 2.0

            self.bracket = True

        #       Second case: A lower function value and derivatives of opposite
        #       sign. The minimum is bracketed. If the cubic step is farther from
        #       stp than the secant step, the cubic step is taken, otherwise the
        #       secant step is taken.

        elif sign < 0:  # case2
            self.case = 2
            theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))
            gamma = s * np.sqrt((theta / s) ** 2 - (gx / s) * (gp / s))
            if stp > stx:
                gamma = -gamma
            p = (gamma - gp) + theta
            q = ((gamma - gp) + gamma) + gx
            r = p / q
            stpc = stp + r * (stx - stp)
            stpq = stp + (gp / (gp - gx)) * (stx - stp)
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            # print (theta, s, gamma, p, q, stpc, stpq, stpf)
            self.bracket = True

        #       Third case: A lower function value, derivatives of the same sign,
        #       and the magnitude of the derivative decreases.

        elif abs(gp) < abs(gx):  # case3
            self.case = 3
            #           The cubic step is computed only if the cubic tends to infinity
            #           in the direction of the step or if the minimum of the cubic
            #           is beyond stp. Otherwise the cubic step is defined to be the
            #           secant step.

            theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))

            #           The case gamma = 0 only arises if the cubic does not tend
            #           to infinity in the direction of the step.
            gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (gx / s) * (gp / s)))
            if stp > stx:
                gamma = -gamma
            p = (gamma - gp) + theta
            q = (gamma + (gx - gp)) + gamma
            r = p / q
            if r < 0.0 and gamma != 0:
                stpc = stp + r * (stx - stp)
            elif stp > stx:
                stpc = stpmax
            else:
                stpc = stpmin
            stpq = stp + (gp / (gp - gx)) * (stx - stp)

            if self.bracket:
                #               A minimizer has been bracketed. If the cubic step is
                #               closer to stp than the secant step, the cubic step is
                #               taken, otherwise the secant step is taken.

                if abs(stpc - stp) < abs(stpq - stp):
                    stpf = stpc
                else:
                    stpf = stpq
                if stp > stx:
                    stpf = min(stp + 0.66 * (sty - stp), stpf)
                else:
                    stpf = max(stp + 0.66 * (sty - stp), stpf)
            else:
                #               A minimizer has not been bracketed. If the cubic step is
                #               farther from stp than the secant step, the cubic step is
                #               taken, otherwise the secant step is taken.

                if abs(stpc - stp) > abs(stpq - stp):
                    stpf = stpc
                else:
                    stpf = stpq
                stpf = min(stpmax, stpf)
                stpf = max(stpmin, stpf)

        #       Fourth case: A lower function value, derivatives of the same sign,
        #       and the magnitude of the derivative does not decrease. If the
        #       minimum is not bracketed, the step is either minstep or maxstep,
        #       otherwise the cubic step is taken.

        else:  # case4
            self.case = 4
            if self.bracket:
                theta = 3.0 * (fp - fy) / (sty - stp) + gy + gp
                s = max(abs(theta), abs(gy), abs(gp))
                gamma = s * np.sqrt((theta / s) ** 2 - (gy / s) * (gp / s))
                if stp > sty:
                    gamma = -gamma
                p = (gamma - gp) + theta
                q = ((gamma - gp) + gamma) + gy
                r = p / q
                stpc = stp + r * (sty - stp)
                stpf = stpc
            elif stp > stx:
                stpf = stpmax
            else:
                stpf = stpmin

        #       Update the interval which contains a minimizer.

        if fp > fx:
            sty = stp
            fy = fp
            gy = gp
        else:
            if sign < 0:
                sty = stx
                fy = fx
                gy = gx
            stx = stp
            fx = fp
            gx = gp
        #       Compute the new step.

        stp = self.determine_step(stpf, old_stp, pk)

        return stx, sty, stp, gx, fx, gy, fy

    def determine_step(self, stp, old_stp, pk):
        dr = stp - old_stp
        x = np.reshape(pk, (-1, 3))
        steplengths = ((dr * x) ** 2).sum(1) ** 0.5
        maxsteplength = max(steplengths)
        if maxsteplength >= self.maxstep:
            dr *= self.maxstep / maxsteplength
        stp = old_stp + dr
        return stp

    def determine_step_(self, pk):
        x = np.reshape(pk, (-1, 3))
        steplengths = ((x) ** 2).sum(1) ** 0.5
        maxsteplength = max(steplengths)
        if maxsteplength >= self.maxstep:
            return self.maxstep / maxsteplength
        else:
            return 1.0

    def save(self, at_idx, data):
        if self.bracket:
            self.isave[at_idx][0] = 1
        else:
            self.isave[at_idx][0] = 0
        self.isave[at_idx][1] = data[0]
        self.dsave[at_idx] = data[1:]

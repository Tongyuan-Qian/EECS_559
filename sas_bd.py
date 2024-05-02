from utils import *


class SasBd:
    def __init__(self, theta, p0, y):
        self.alph = 0
        self.max_iter = 1000
        self.eps = 1e-3
        self.back_track = np.array([0.1, 0.1])
        self.refine_iter = np.outer(np.array([1, 1, 1, 1, 1]), np.array([10, 20]))
        self.lbd = 5e-1 * 1 / np.sqrt(p0 * theta)

        self.y = y
        self.y_f = fft(y)
        self.y_norm = norm(y)
        self.n = len(y)
        self.p0 = p0
        self.a = self.init_a()
        self.obj_list = np.zeros(self.max_iter)
        self.x = np.zeros(self.n)

    def init_a(self):
        # truncate y
        idx = (np.random.randint(0, self.n) + np.arange(self.p0)) % self.n
        a = self.y[idx]
        # pad a
        a = np.pad(a, (self.p0 - 1, self.p0 - 1), "constant")
        a = a / norm(a)
        # grad
        a, _ = self.grad_f(a)
        # normalize
        a = -a / norm(a)

        return a

    def grad_f(self, a, lbd=0.1):
        x = real(ifft(self.y_f * conj(fft(a, self.n))))
        x = soft(x, lbd)
        grad = -real(ifft(self.y_f * conj(fft(x))))
        return grad[0:len(a)], x

    def min_obj_f(self, x):
        return self.y_norm ** 2 / 2 - norm(x) ** 2 / 2

    def ref_x_obj_f(self, x, a, lbd):
        return (1 / 2) * norm(real(ifft(fft(x) * fft(a, self.n))) - self.y) ** 2 + lbd * norm(x, 1)

    def ref_a_obj_f(self, x, a):
        return (1 / 2) * norm(real(ifft(fft(x) * fft(a, self.n))) - self.y) ** 2

    def hess_f(self, ):
        ...

    def solve(self):
        # minimization
        self.minimization()

        # refinement
        self.refinement()

        return self.x, self.a

    def minimization(self):
        for itr in range(self.max_iter):
            w = self.a
            e_grad, x = self.grad_f(w, self.lbd)
            r_grad = e2r_grad(w, e_grad)
            r_grad_norm_sq = norm(r_grad) ** 2
            obj_ = self.min_obj_f(x)

            t = 1
            while True:
                a = exp_f(w, -t * r_grad)
                _, x = self.grad_f(a, self.lbd)
                obj = self.min_obj_f(x)

                if obj - obj_ >= - self.back_track[1] * t * r_grad_norm_sq:
                    t = t * self.back_track[0]
                else:
                    break

            self.a = a
            self.x = x
            self.obj_list[itr] = obj
            if itr > 0 and t * r_grad_norm_sq < self.eps:
                break

    def refinement(self):
        lbd = 1
        for itr in range(self.refine_iter.shape[0]):
            # update x
            self.x = self.refine_x(self.refine_iter[itr, 0], self.x, self.a, lbd)
            # update a
            self.a = self.refine_a(self.refine_iter[itr, 1], self.x, self.a)
            # update param
            lbd /= 2

    def refine_x(self, max_iter, x, a, lbd, eps=1e-4):
        t_ = 1
        w_f = fft(x)
        x_f_ = w_f
        a_f = fft(a, self.n)
        aa_f = conj(a_f) * a_f
        ay_f = conj(a_f) * self.y_f
        gam = 0.99 / np.max(real(aa_f))

        obj_list = np.zeros(max_iter)
        for itr in range(max_iter):
            x = soft(real(ifft(w_f - gam * (aa_f * w_f - ay_f))), gam * lbd)
            t = (1 + np.sqrt(1 + 4 * t_ ** 2)) / 2
            x_f = fft(x)
            w_f = x_f + (t_ - 1) / t * (x_f - x_f_)

            x_f_ = x_f
            t_ = t

            obj_list[itr] = self.ref_x_obj_f(x, a, lbd)
            if itr > 0 and abs(obj_list[itr] - obj_list[itr - 1]) < eps:
                break

        return x

    def refine_a(self, max_iter, x, a, eps=1e-4):
        a_ = a
        y_supp = np.abs(c_conv(np.where(x != 0, 1, 0), np.ones(len(a_)), self.n)) > eps
        x_f = fft(x)
        gam = 1 / np.max(np.abs(x_f)) ** 2

        obj_list = np.zeros(max_iter)
        for itr in range(max_iter):
            a_grad = a + 0.9 * (a - a_)
            a_grad = real(ifft(x_f * fft(a_grad, self.n))) - self.y
            a_grad = real(ifft(conj(x_f) * fft(y_supp * a_grad)))
            a_grad = a_grad[:len(a_)]

            a_ = a
            a = a - gam * a_grad

            obj_list[itr] = self.ref_a_obj_f(x, a)
            if itr > 0 and abs(obj_list[itr] - obj_list[itr - 1]) < eps:
                break
        return a / norm(a)

z18 = """
 ЗАДАНИЕ 18. Моделирование вычислительной системы с удаленными терминалами 
 Вычислительная система представляет собой двухпроцессорный комплекс, который 
 обслуживает местных пользователей и три однотипных удаленных терминала. 
 На каждом терминале задача формируется в среднем через t сек, а время выполнения  
 задачи в процессорном блоке имеет математическое ожидание µ  сек (экспоненциальное 
 распределение). После выполнения задача возвращается на соответствующий терминал, 
 инициируя  тем самым  формирование новой задачи. Время передачи данных по каналу связи
 распределяется равномерно в пределах от a до b сек.  

Разработать GPSSV- модель для анализа процесса функционирования вычислительной системы  в течение одного часа.
Первоначальный перечень экспериментов: t=25, µ=20, a=10, b=30. 
"""

import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

debug = False
def P(*args):
    if debug: print(*args)
    pass


class Lab18(object):

    class Terminal(object):
        """Represents a single terminal for users."""

        def __init__(self, obj, env, resource, t=25, mu=20, name=""):
            self.obj = obj
            self.env = env
            self.resource = resource
            self.action = env.process(self.run())
            self.t = t
            self.mu = mu
            self.name = name
            pass

        def P(self, *args):
            P(self.name,*args)
            pass

        def run(self):
            """Running process."""
            while True:
                # Generate problem
                st = self.env.now
                yield self.env.process(self.gen_problem())
                self.obj.gen_times.append(self.env.now - st)

                # Solution
                st = self.env.now
                yield self.env.process(self.request_solve())
                self.obj.solve_times.append(self.env.now - st)
                pass
            pass

        def gen_problem(self):
            """Simulate generating a problem."""
        
            self.P("@ %04.02f Generating problem " % self.env.now)
            duration = random.expovariate(1.0/self.t)
            yield self.env.timeout(duration) # Generate next problem
            pass

        def request_solve(self):
            """Simulate a request to solve the problem."""
        
            with self.resource.request() as req:
                self.P("@ %04.02f Requesting terminal access " % self.env.now)
                yield req # Request access

                self.P("@ %04.02f Requesting solution" % self.env.now)
                duration = random.expovariate(1.0/self.mu)
                yield self.env.timeout(duration) # Do calculation
            pass
        pass

    class RemoteTerminal(Terminal):
        """Represents a remote user."""

        def __init__(self, obj, env, resource, t = 25, mu = 20, a=10, b=30, name=""):
            super().__init__(obj, env, resource, t=t, mu=mu, name=name)
            self.a = a
            self.b = b
            pass

        def request_solve(self):
            """Simulate a request to solve the problem.
            Note that extra time is required to transfer data."""

            with self.resource.request() as req:
                self.P("@ %04.02f Requesting terminal access" % self.env.now)
                yield req # Request access

                self.P("@ %04.02f Moving data" % self.env.now)
                transf_time = random.randint(self.a, self.b)
                yield self.env.timeout(transf_time) # Transfer data

                self.P("@ %04.02f Requesting solution" % self.env.now)
                duration = random.expovariate(1.0/self.mu)
                yield self.env.timeout(duration) # Do calculation
            pass
        pass

    def __init__(self, t = 25, mu = 20, a=10, b=30, n_remote=3, n_processors=2):
        # Options
        self.a = a
        self.b = b
        self.t = t
        self.mu = mu
        self.n_remote = 3
        self.n_processors = n_processors

        # Save data
        self.gen_times = []
        self.solve_times = []
        pass

    def run(self, T=3600):
        """Runs a simulation for T seconds."""
        env = simpy.Environment()
        procs = simpy.Resource(env, capacity=self.n_processors)
        
        home_term = Lab18.Terminal(self, env, procs, name="home  ")
        remote_terms = [Lab18.RemoteTerminal(self, env, procs, name="sess%d " % i) for i in range(3)]

        env.run(until=3600) # Run for T = 60*60 seconds = 60 minutes
        return self

    def output(self):
        plt.title("Problem generation times.")
        plt.hist(self.gen_times)
        plt.show()

        plt.title("Problem solution times (incl. waiting).")
        plt.hist(self.solve_times)
        plt.show()
        return self
    pass


debug = True

l18 = Lab18().run(3600)

print("Average gen time: %0.2f" % np.mean(l18.gen_times))
print("Average solve time: %0.2f" % np.mean(l18.solve_times))

l18.output()


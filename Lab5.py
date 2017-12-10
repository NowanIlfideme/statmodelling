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

gen_times = []
solve_times = []

class Terminal(object):

    def __init__(self, env, resource, t=25, mu=20, name=""):
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
            gen_times.append(self.env.now - st)

            # Solution
            st = self.env.now
            yield self.env.process(self.request_solve())
            solve_times.append(self.env.now - st)
            pass
        pass

    def gen_problem(self):
        """Simulate generating a problem."""
        
        self.P("Generating problem @ %f.02" % self.env.now)
        duration = random.expovariate(1.0/self.t)
        yield self.env.timeout(duration) # Generate next problem
        pass

    def request_solve(self):
        """Simulate a request to solve the problem."""
        
        with self.resource.request() as req:
            self.P("Requesting terminal access @ %f.02" % self.env.now)
            yield req # Request access

            self.P("Requesting solution @ %f.02" % self.env.now)
            duration = random.expovariate(1.0/self.mu)
            yield self.env.timeout(duration) # Do calculation
        pass
    pass

class RemoteTerminal(Terminal):
    
    def __init__(self, env, resource, t = 25, mu = 20, a=10, b=30, name=""):
        super().__init__(env, resource, t=t, mu=mu, name=name)
        self.a = a
        self.b = b
        pass

    def request_solve(self):
        """Simulate a request to solve the problem.
        Note that extra time is required to transfer data."""

        with self.resource.request() as req:
            self.P("Requesting terminal access @ %f.02" % self.env.now)
            yield req # Request access

            self.P("Moving data @ %f.02" % self.env.now)
            transf_time = random.randint(self.a, self.b)
            yield self.env.timeout(transf_time) # Transfer data

            self.P("Requesting solution @ %f.02" % self.env.now)
            duration = random.expovariate(1.0/self.mu)
            yield self.env.timeout(duration) # Do calculation
        pass
    pass

env = simpy.Environment()
procs = simpy.Resource(env, capacity=2)
home_term = Terminal(env, procs, name="home  ")
remote_terms = [RemoteTerminal(env, procs, name="sess%d " % i) for i in range(3)]
env.run(until=3600) # Run for T = 60*60 seconds = 60 minutes

gen_times = np.array(gen_times)
solve_times = np.array(solve_times)

plt.title("Problem generation times.")
plt.hist(gen_times)
plt.show()

plt.title("Problem solution times (incl. waiting).")
plt.hist(solve_times)
plt.show()


z37 = """
 ЗАДАНИЕ 37. Моделирование процесса функционирования участка контроля 
 Изделия поступают из цеха на контроль через каждые а±δ мин. (здесь и далее равномерный закон). 
 Каждый из 2-х контролеров выполняет свои функции  мин. После контроля примерно 75% изделий 
 направляется на склад, а остальные – к наладчику для доводки. Наладчик выполняет свои функции b±β c±γ мин.,
 и возвращает изделия на повторный контроль. На участке контроля фиксируется число изделий, 
 направленных на склад и прошедших наладчика. После окончания смены изделия из цеха не поступают, 
 а изделия из очереди обслуживаются контролерами.
 
Разработать GPSSV – модель для анализа процесса функционирования участка контроля в течение смены, т.е. 8 часов. 
Первоначальный перечень экспериментов: a=5, δ=2, b=9, β=3, c=30, γ=10.
"""
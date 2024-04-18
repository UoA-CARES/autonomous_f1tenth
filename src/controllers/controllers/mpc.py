import rclpy
from rclpy import Future
from rclpy.node import Node
import numpy as np
import do_mpc
from casadi import *
import matplotlib.pyplot as plt
import matplotlib as mpl


class MPC():
    def __init__(self, alg): 
        self.alg = alg
        self.deltaT = 0.1
        self.wheelbase = 0.315
        self.timeConst = 0.1
        self.predictionSteps = 2

        
    
    def newStates(self, lin, x, y, steeringAngle, desAngle, yaw):
        # Uses simple bicycle model
        x = x + lin*np.cos(yaw)*self.deltaT
        y = y + lin*np.sin(yaw)*self.deltaT
        yaw = yaw + self.deltaT*(lin*np.tan(steeringAngle))/self.wheelbase
        steeringAngle = steeringAngle - self.timeConst**(-1)*(steeringAngle - desAngle)*self.deltaT

        return x, y, yaw, steeringAngle
    
    def cost(self, xcurr, x, xdes, ycurr, y, ydes, steeringcurr, steering, yawcurr, yaw, yawdes):
        Y = np.array([[xcurr, ycurr, yawcurr, 0, x, y, yaw, 0]])
        Yref = np.array([[xdes, ydes, yawdes, 0, xdes, ydes, yawdes, 0]])
        Yarr = Y - Yref
        qx = 10
        qy = 100
        qyaw = 1000
        Q = np.diag([qx, qy, qyaw, 0 , qx, qy, qyaw, 0])
        qsteer = 50

        cost = Yarr@Q@np.transpose(Yarr) + qsteer*(steeringcurr - steering)**2 #(U - Uref)*R*(U-Uref)
        return cost
    
    def mpcAlg(self, x, y, steering, des_angle, lin, yaw):
        MAX_ACTIONS = np.asarray([0.5, 0.85])
        MIN_ACTIONS = np.asarray([0, -0.85])
        """

        # Plotting config
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.grid'] = True

        # Model config
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # Set variables

        # Confirm variables
        model.setup()

        # MPC Controller config
        mpc = do_mpc.controller.MPC(model)

        setup_mpc = {
            'n_horizon': 20,
            't_step': 0.1,
            'n_robust': 1,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)
        
        """
        """# Customizing Matplotlib:
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.grid'] = True

        xcurr = 7
        x = 5
        ycurr = 2
        y = 1
        yawcurr = 8
        yaw = 3
        xdes = 0.1
        ydes = 0.5
        yawdes = 0.8

        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        phi_1 = model.set_variable(var_type='_x', var_name='phi_1', shape=(1,1))
        phi_2 = model.set_variable(var_type='_x', var_name='phi_2', shape=(1,1))
        phi_3 = model.set_variable(var_type='_x', var_name='phi_3', shape=(1,1))
        # Variables can also be vectors:
        dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(3,1))
        # Two states for the desired (set) motor position:
        phi_m_1_set = model.set_variable(var_type='_u', var_name='phi_m_1_set')
        phi_m_2_set = model.set_variable(var_type='_u', var_name='phi_m_2_set')
        # Two additional states for the true motor position:
        phi_1_m = model.set_variable(var_type='_x', var_name='phi_1_m', shape=(1,1))
        phi_2_m = model.set_variable(var_type='_x', var_name='phi_2_m', shape=(1,1))

        # As shown in the table above, we can use Long names or short names for the variable type.
        Theta_1 = model.set_variable('parameter', 'Theta_1')
        Theta_2 = model.set_variable('parameter', 'Theta_2')
        Theta_3 = model.set_variable('parameter', 'Theta_3')

        c = np.array([2.697,  2.66,  3.05, 2.86])*1e-3
        d = np.array([6.78,  8.01,  8.82])*1e-5

        model.set_rhs('phi_1', dphi[0])
        model.set_rhs('phi_2', dphi[1])
        model.set_rhs('phi_3', dphi[2])

        dphi_next = vertcat(
            -c[0]/Theta_1*(phi_1-phi_1_m)-c[1]/Theta_1*(phi_1-phi_2)-d[0]/Theta_1*dphi[0],
            -c[1]/Theta_2*(phi_2-phi_1)-c[2]/Theta_2*(phi_2-phi_3)-d[1]/Theta_2*dphi[1],
            -c[2]/Theta_3*(phi_3-phi_2)-c[3]/Theta_3*(phi_3-phi_2_m)-d[2]/Theta_3*dphi[2],
        )

        model.set_rhs('dphi', dphi_next)

        tau = 1e-2
        model.set_rhs('phi_1_m', 1/tau*(phi_m_1_set - phi_1_m))
        model.set_rhs('phi_2_m', 1/tau*(phi_m_2_set - phi_2_m))

        model.setup()

        mpc = do_mpc.controller.MPC(model)

        setup_mpc = {
            'n_horizon': 20,
            't_step': 0.1,
            'n_robust': 1,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)

        mterm = phi_1**2 + phi_2**2 + phi_3**2
        lterm = phi_1**2 + phi_2**2 + phi_3**2

        mpc.set_objective(mterm=mterm, lterm=lterm)

        mpc.set_rterm(
            phi_m_1_set=1e-2,
            phi_m_2_set=1e-2
        )

        # Lower bounds on states:
        mpc.bounds['lower','_x', 'phi_1'] = -2*np.pi
        mpc.bounds['lower','_x', 'phi_2'] = -2*np.pi
        mpc.bounds['lower','_x', 'phi_3'] = -2*np.pi
        # Upper bounds on states
        mpc.bounds['upper','_x', 'phi_1'] = 2*np.pi
        mpc.bounds['upper','_x', 'phi_2'] = 2*np.pi
        mpc.bounds['upper','_x', 'phi_3'] = 2*np.pi

        # Lower bounds on inputs:
        mpc.bounds['lower','_u', 'phi_m_1_set'] = -2*np.pi
        mpc.bounds['lower','_u', 'phi_m_2_set'] = -2*np.pi
        # Lower bounds on inputs:
        mpc.bounds['upper','_u', 'phi_m_1_set'] = 2*np.pi
        mpc.bounds['upper','_u', 'phi_m_2_set'] = 2*np.pi

        mpc.scaling['_x', 'phi_1'] = 2
        mpc.scaling['_x', 'phi_2'] = 2
        mpc.scaling['_x', 'phi_3'] = 2

        inertia_mass_1 = 2.25*1e-4*np.array([1., 0.9, 1.1])
        inertia_mass_2 = 2.25*1e-4*np.array([1., 0.9, 1.1])
        inertia_mass_3 = 2.25*1e-4*np.array([1.])

        mpc.set_uncertainty_values(
            Theta_1 = inertia_mass_1,
            Theta_2 = inertia_mass_2,
            Theta_3 = inertia_mass_3
        )


        mpc.setup()

        simulator = do_mpc.simulator.Simulator(model)

        # Instead of supplying a dict with the splat operator (**), as with the optimizer.set_param(),
        # we can also use keywords (and call the method multiple times, if necessary):
        simulator.set_param(t_step = 0.1)

        p_template = simulator.get_p_template()

        def p_fun(t_now):
            p_template['Theta_1'] = 2.25e-4
            p_template['Theta_2'] = 2.25e-4
            p_template['Theta_3'] = 2.25e-4
            return p_template

        simulator.set_p_fun(p_fun)

        simulator.setup()

        x0 = np.pi*np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1,1)

        simulator.x0 = x0
        mpc.x0 = x0

        mpc.set_initial_guess()

        mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
        sim_graphics = do_mpc.graphics.Graphics(simulator.data)


        # We just want to create the plot and not show it right now. This "inline magic" supresses the output.
        fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
        fig.align_ylabels()
        fig


        for g in [sim_graphics, mpc_graphics]:
            # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
            g.add_line(var_type='_x', var_name='phi_1', axis=ax[0])
            g.add_line(var_type='_x', var_name='phi_2', axis=ax[0])
            g.add_line(var_type='_x', var_name='phi_3', axis=ax[0])

            # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
            g.add_line(var_type='_u', var_name='phi_m_1_set', axis=ax[1])
            g.add_line(var_type='_u', var_name='phi_m_2_set', axis=ax[1])


        ax[0].set_ylabel('angle position [rad]')
        ax[1].set_ylabel('motor angle [rad]')
        ax[1].set_xlabel('time [s]')

        u0 = np.zeros((2,1))
        for i in range(200):
            simulator.make_step(u0)
            

        sim_graphics.plot_results()
        # Reset the limits on all axes in graphic to show the data.
        sim_graphics.reset_axes()
        # Show the figure:
        plt.show()

        u0 = mpc.make_step(x0)


        # #Y = np.array([[xcurr, x, ycurr, 0, y, yawcurr, yaw, 0]])
        # Y = np.array([[xcurr, ycurr, yawcurr, 0, x, y, yaw, 0]])
        # Yref = np.array([[xdes, ydes, yawdes, 0, xdes, ydes, yawdes, 0]])
        # #Yref = np.array([[xdes, xdes, ydes, 0, ydes, yawdes, yawdes, 0]])
        # Yarr = Y - Yref

        # print(Yarr)
        # print(np.transpose(Yarr))
        # qx = 10
        # qy = 100
        # qyaw = 1000
        # Q = np.diag([qx, qy, qyaw, 0 , qx, qy, qyaw, 0])
        # print(Q)

        # J = Yarr@Q@np.transpose(Yarr)
        # print(J)"""


        return lin, steering
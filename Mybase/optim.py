import numpy as np
import tensorflow as tf

def exponential_decay(config = None, global_step = None):
    
    #assert cofig['decay_rule'] == 'exponential_decay', "Optim rule error!"
    
    lr_base    = config.get('lr_base', 1e-2)
    decay_rate = config.get('decay_rate', 0.9)
    decay_step = config.get('decay_step', 100)
    staircase  = config.get('staircase', True)
    
    return tf.train.exponential_decay(lr_base, global_step, decay_step, decay_rate, staircase=staircase)


def fixed_decay(config = None, global_step = None):
    """
    boundaries = [decay_step*x for x in np.array([82, 123, 300], dtype=np.int64)]
    staged_lr = [lr_base*x for x in [1, 0.1, 0.01, 0.002]]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, staged_lr)
    #learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(), boundaries, staged_lr)
    """
    
    lr_base = config.get('lr_base', 1e-2)
    
    return tf.constant(lr_base)
    

def polynomial_decay(config = None, global_step = None):
    
    lr_base = config.get('lr_base', 1e-2)
    lr_end  = config.get('lr_end', 1e-4)
    decay_step  = config.get('decay_step', 100)
    power = config.get('power', 0.9)
    cycle = config.get('cycle', False)
    
    return tf.train.polynomial_decay(lr_base, global_step, decay_step, lr_end, power=power, cycle=cycle)


def sgd_optim(config = None, global_step = None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
      lr_base: Scalar learning rate.
    """
    learning_rate = config["learning_rate"]
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate)
    return train_step


def momentum_optim(config = None, global_step = None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
    
    Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
                average of the gradients.
    """
    learning_rate = config["learning_rate"]
    
    momentum = config.get('momentum', 0.9)
    
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum)

    return train_step


def rmsprop_optim(config = None, global_step = None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    learning_rate = config["learning_rate"]

    decay      = config.get('decay', 0.9)
    momentum   = config.get('momentum', 0.99)
    epsilon    = config.get('epsilon', 1e-8)
    
    train_step = tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon)

    return train_step


def adam_optim(config = None, global_step = None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    learning_rate = config["learning_rate"]

    beta1   = config.get('beta1', 0.9)
    beta2   = config.get('beta2', 0.999)
    epsilon = config.get('epsilon', 1e-8)
        
    train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon)

    return train_step


def adagrad_optim(config = None, global_step = None):
    """
    Args:
        learning_rate: A Tensor or a floating point value. The learning rate.
        initial_accumulator_value: A floating point value. Starting value for the accumulators, must be positive.
        use_locking: If True use locks for update operations.
    
    Returns:
        An instance of an optimizer.
    """
    learning_rate = config["learning_rate"]
    
    init_acc_value = config.get('init_acc_value', 0.1)
        
    train_step = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=init_acc_value)
    
    return train_step


def adadelta_optim(config = None, global_step = None):
    """
    Args:
        learning_rate: A Tensor or a floating point value. The learning rate. To match the exact form in the original paper use 1.0.
        rho: A Tensor or a floating point value. The decay rate.
        epsilon: A Tensor or a floating point value. A constant epsilon used to better conditioning the grad update.
        use_locking: If True use locks for update operations.
    
    Returns:
        An instance of an optimizer.
    """
    learning_rate = config["learning_rate"]
    
    rho = config.get('rho', 0.95)
    epsilon = config.get('epsilon', 1e-08)

    train_step = tf.train.AdadeltaOptimizer(learning_rate, rho=rho, epsilon=epsilon)
    
    return train_step


def ftrl_optim(config = None, global_step = None):
    """
    Args:
        learning_rate: A float value or a constant float Tensor.
        learning_rate_power: A float value, must be less or equal to zero.
        initial_accumulator_value: The starting value for accumulators. Only positive values are allowed.
        l1_regularization_strength: A float value, must be greater than or equal to zero.
        l2_regularization_strength: A float value, must be greater than or equal to zero.
        use_locking: If True use locks for update operations.
        name: Optional name prefix for the operations created when applying gradients. Defaults to "Ftrl".
        accum_name: The suffix for the variable that keeps the gradient squared accumulator. If not present, defaults to name.
        linear_name: The suffix for the variable that keeps the linear gradient accumulator. If not present, defaults to name + "_1".
        l2_shrinkage_regularization_strength: A float value, must be greater than or equal to zero. 
            This differs from L2 above in that the L2 above is a stabilization penalty, whereas 
            this L2 shrinkage is a magnitude penalty. The FTRL formulation can be written as: 
            w_{t+1} = argmin_w(\hat{g}{1:t}w + L1||w||_1 + L2||w||_2^2), where \hat{g} = g + (2L2_shrinkagew), 
            and g is the gradient of the loss function w.r.t. the weights w. Specifically, in the absence of 
            L1 regularization, it is equivalent to the following update rule: 
            w{t+1} = w_t - lr_t / (1 + 2L2lr_t) * g_t - 2L2_shrinkagelr_t / (1 + 2L2lr_t) * w_t
            where lr_t is the learning rate at t. When input is sparse shrinkage will only happen on the active weights.

    Returns:
        An instance of an optimizer.
    """
    
    learning_rate = config["learning_rate"]
    
    lr_power = config.get('lr_power', -0.5)
    init_acc_value = config.get('init_acc_value', 0.1)
    ftrl_l1 = config.get('ftrl_l1', 0.0)
    ftrl_l2 = config.get('ftrl_l2', 0.0)
        
    train_step = tf.train.FtrlOptimizer(learning_rate, learning_rate_power=lr_power, \
                                        initial_accumulator_value=init_acc_value,
                                        l1_regularization_strength=ftrl_l1,
                                        l2_regularization_strength=ftrl_l2)
    
    return train_step
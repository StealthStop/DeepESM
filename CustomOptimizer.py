from keras.optimizers import Adam
from tensorflow.python.training import training_ops
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.framework import ops
from tensorflow.python.distribute import values as ds_values

class CustomAdam(Adam):

    # This wrapper to the adam, optimizer_v2 classes keeps track of different learning rates
    # and decay parameters for specified parts of a neural network model
    # Thus, an lr, beta_1, and beta_2 are provided for the mass regression component
    # and then for disc components. Default values are also provided for all other components
    def __init__(self, mass_reg_lr=1.0, disc_lr=0.001, other_lr=0.001,
                       mass_reg_beta_1=0.9, mass_reg_beta_2=0.999, disc_beta_1=0.9, disc_beta_2=0.999, 
                       other_beta_1=0.9, other_beta_2=0.999, epsilon=1e-8, amsgrad=False, **kwargs):

        super().__init__(name = "CustomAdam", **kwargs)

        # Set Adam hyperparameters separately for the mass regession layers, disc layers, and all others (default)
        self._set_hyper('mass_reg_lr', mass_reg_lr)
        self._set_hyper('mass_reg_beta_1', mass_reg_beta_1)
        self._set_hyper('mass_reg_beta_2', mass_reg_beta_2)

        self._set_hyper('disc_lr', disc_lr)
        self._set_hyper('disc_beta_1', disc_beta_1)
        self._set_hyper('disc_beta_2', disc_beta_2)

        self._set_hyper("other_lr", other_lr)
        self._set_hyper('other_beta_1', other_beta_1)
        self._set_hyper('other_beta_2', other_beta_2)

        self.epsilon = epsilon
        self.amsgrad = amsgrad

    # The resource apply method is responsible for applying the gradient
    # and using the current state of the learning rate-related variables
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var.name, var_device, var_dtype))
                        or self._prepare_local(var.name, var_device, var_dtype, apply_state))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        if not self.amsgrad:
            return training_ops.resource_apply_adam(
                var.handle,
                m.handle,
                v.handle,
                coefficients['beta_1_power'],
                coefficients['beta_2_power'],
                coefficients['lr'],
                coefficients['beta_1_t'],
                coefficients['beta_2_t'],
                coefficients['epsilon'],
                grad,
                use_locking=self._use_locking)
        else:
           vhat = self.get_slot(var, 'vhat')
           return training_ops.resource_apply_adam_with_amsgrad(
               var.handle,
               m.handle,
               v.handle,
               vhat.handle,
               coefficients['beta_1_power'],
               coefficients['beta_2_power'],
               coefficients['lr'],
               coefficients['beta_1_t'],
               coefficients['beta_2_t'],
               coefficients['epsilon'],
               grad,
               use_locking=self._use_locking)

    # This overridden method makes a dictionary "apply_state" that maps
    # a model variable tuple to a dictionary that will contain
    # information related to the learning rate-related hyperparmeters.
    # E.g. each layer (kernel/bias/etc) specified in the model will appear in the var list
    # therefore, the dictionary maps the unique layer name, GPU device, and variable type
    # to the learning rate hyperparams to keep track of them as Adam updates/uses them
    # during training.
    def _prepare(self, var_list):
        keys = set()
        for var in var_list:
            if isinstance(var, ds_values.DistributedValues):
                var_devices = var._devices
            else:
                var_devices = [var.device]
            var_dtype = var.dtype.base_dtype
            for var_device in var_devices:
                keys.add((var.name, var_device, var_dtype))

        apply_state = {}
        for var, var_device, var_dtype in keys:
            apply_state[(var, var_device, var_dtype)] = {}
            with ops.device(var_device):
                 self._prepare_local(var, var_device, var_dtype, apply_state)

        return apply_state

    # This overridden adam, optimizer_v2 method updates the hyperparams
    # for a given layer based on the current state of the learning
    # Importantly, the learning rate is adjusted based on Adam's decay mechanism.
    def _prepare_local(self, var_name, var_device, var_dtype, apply_state):

        # Figure out which layer/device we are processing
        # To decide which hyperparamters to use and update
        splits = var_name.split("_")
        component = "_".join(splits[:-1])

        if component not in ["disc", "mass_reg"]:
            component = "other"

        local_step   = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t     = array_ops.identity(self._get_hyper('%s_beta_1'%(component), var_dtype))
        beta_2_t     = array_ops.identity(self._get_hyper('%s_beta_2'%(component), var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr           = (self._get_hyper('%s_lr'%(component), var_dtype) *
                       (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))

        apply_state[(var_name, var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

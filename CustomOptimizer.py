from keras.optimizers import Adam
from tensorflow.python.training import training_ops

class CustomAdam(Adam):

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., amsgrad=False,
                 config=None, **kwargs):

        super().__init__(name = "CustomAdam", **kwargs)

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

        self.config = config

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        theLR = None
        for name, lr in self.config.items():
            splits = name.split("_")
            if splits[-1] != "lr": continue

            elif "_".join(splits[:-1]) in var.name:
                print(name, lr)
                theLR = lr 
                break
            else:
                theLR = self.config["default_lr"]

        print(var.name, theLR)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        if not self.amsgrad:
          return training_ops.resource_apply_adam(
              var.handle,
              m.handle,
              v.handle,
              coefficients['beta_1_power'],
              coefficients['beta_2_power'],
              theLR,
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
              theLR,
              coefficients['beta_1_t'],
              coefficients['beta_2_t'],
              coefficients['epsilon'],
              grad,
              use_locking=self._use_locking)

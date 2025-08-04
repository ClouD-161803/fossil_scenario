"""
This module defines the Certificate class and its subclasses, which are used to guide
the learner and verifier components in the fossil library. Certificates are used to 
certify properties of a system, such as stability or safety. The module also defines 
functions for logging loss and accuracy during the learning process, and for checking 
that the domains and data are as expected for a given certificate.
"""
# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Generator, Type, Any, Union

import torch
import copy
import numpy as np
from torch.optim import Optimizer

import fossil.control as control
import fossil.logger as logger
import fossil.learner as learner
from fossil.consts import ScenAppConfig, CertificateType, DomainNames, ScenAppStateKeys
import fossil.domains as domains

torch.set_num_threads(8)
XD = DomainNames.XD.value
XD1 = DomainNames.XD1.value
XD2 = DomainNames.XD2.value
XI = DomainNames.XI.value
XU = DomainNames.XU.value
XS = DomainNames.XS.value
XG = DomainNames.XG.value
XG1 = DomainNames.XG1.value
XG2 = DomainNames.XG2.value
XG_BORDER = DomainNames.XG_BORDER.value
XG1_BORDER = DomainNames.XG1_BORDER.value
XG2_BORDER = DomainNames.XG2_BORDER.value
XS_BORDER = DomainNames.XS_BORDER.value
XF = DomainNames.XF.value
XNF = DomainNames.XNF.value
XR = DomainNames.XR.value  # This is an override data set for ROA in StableSafe
HAS_BORDER = (XG, XS)
BORDERS = (XG_BORDER, XS_BORDER)
ORDER = (XD, XI, XU, XS, XG, XG_BORDER, XS_BORDER, XF, XNF)

cert_log = logger.Logger.setup_logger(__name__)


def safe_set_beta(learner_obj, beta_value):
    """Safely set the beta attribute on a learner object if it has that attribute.
    
    Args:
        learner_obj: The learner object that may have a beta attribute
        beta_value: The value to assign to the beta attribute
        
    Returns:
        The learner object with beta potentially updated
    """
    if hasattr(learner_obj, 'beta'):
        if isinstance(beta_value, torch.Tensor):
            beta_value = beta_value.item()
        learner_obj.beta = beta_value
    return learner_obj


def log_loss_acc(t, loss, accuracy, verbose):
    # cert_log.debug(t, "- loss:", loss.item())
    # for k, v in accuracy.items():
    #     cert_log.debug(" - {}: {}%".format(k, v))
    loss_value = loss.item() if hasattr(loss, "item") else loss
    cert_log.debug("{} - loss: {:.5f}".format(t, loss_value))

    for k, v in accuracy.items():
        cert_log.debug(" - {}: {:.3f}%".format(k, v))


def _set_assertion(required, actual, name):
    if required != actual:
        raise ValueError(
            "Required {} {} do not match actual domains {}. Missing: {}, Not required: {}".format(
                name, required, actual, required - actual, actual - required
            )
        )


class Certificate:
    """
    Base class for certificates, used to define new Certificates.
    Certificates are used to guide the learner and verifier components.
    Methods learn and get_constraints must be implemented by subclasses.

    Attributes:
        domains: (symbolic) domains of the system. This is a dictionary of domain names and symbolic domains as SMT
            formulae.
            These may be stored as separate attributes for each domain, or
            as a dictionary of domain names and domains. They should be accessed accordingly.
        bias: Should the network have bias terms for this certificate? (default: True)
        max_jumps: Maximum number of jumps/compression set size (default: -1, no limit)
        use_apriori_jumps: Whether to use the a priori jump limit algorithm (default: False)
    """

    bias = True
    max_jumps = -1
    use_apriori_jumps = False

    def __init__(self, domains: dict[str, Any], config: Union[ScenAppConfig, None] = None) -> None:
        if config is not None:
            self.max_jumps = getattr(config, 'MAX_JUMPS', -1)
            self.use_apriori_jumps = getattr(config, 'USE_APRIORI_JUMPS', False)
        pass

    def get_violations(self, certificate, certificate_dot, S, Sdot, times, state_data) -> tuple[int, int]:
        return 0, 0

    def estimate_beta(self, learner):
        """
        Estimate beta parameter for this certificate type.
        This is a default implementation that returns None.
        Subclasses should override this method to provide specific beta estimation.
        """
        return None
        
    def subsurface_algorithm(
        self,
        losses: dict,
        supp_loss: Union[torch.Tensor, int],
        supp_samples: set,
        best_loss: float,
        learner,
        optimizer,
        discrete: bool = False,
        beta: Union[torch.Tensor, float, None] = None
    ) -> tuple[bool, bool, set, float, Union[learner.LearnerNN, None]]:
        """
        Centralized implementation of the subsurface algorithm used across certificate types.
        
        Args:
            losses: Dictionary mapping sample indices to their loss values
            supp_loss: Loss for the support set or -1 if not defined
            supp_samples: Set of indices in the support set
            best_loss: Current best loss value
            learner: Neural network learner
            optimizer: Optimizer used for training
            discrete: Whether the system is discrete-time
            beta: Beta parameter for certificates that use it
            
        Returns:
            tuple containing:
                - break_flag: Whether to break the training loop
                - new_supp_added: Whether a new sample was added to the support set
                - updated_supp_samples: Updated support sample set
                - updated_best_loss: Updated best loss value
                - updated_best_net: Updated best network (if improved)
        """
        break_flag = False
        new_supp_added = False
        updated_best_loss = best_loss
        updated_best_net = None
        

        max_jumps_reached = hasattr(self, 'use_apriori_jumps') and self.use_apriori_jumps and \
                            hasattr(self, 'max_jumps') and self.max_jumps > 0 and \
                            len(supp_samples) >= self.max_jumps
        
        sorted_keys = sorted(losses, key=lambda k: losses[k], reverse=True)
        max_loss = losses[sorted_keys[0]]
        
        if supp_loss != -1:
            supp_loss_float = supp_loss.item() if isinstance(supp_loss, torch.Tensor) else float(supp_loss)
            if supp_loss_float < best_loss:
                updated_best_loss = supp_loss_float
                updated_best_net = copy.deepcopy(learner)
                if beta is not None:
                    updated_best_net = safe_set_beta(updated_best_net, beta)
            
            optimizer.zero_grad()
            
            # Line 17: If (supp_loss - best_loss) >= η, add new sample to C
            if (supp_loss_float - best_loss) >= 1e-1:
                if sorted_keys[0] in supp_samples or max_jumps_reached:
                    break_flag = True
                else:
                    # Line 19: C ← C ∪ {ξ̄}
                    supp_samples.add(sorted_keys[0])
                    new_supp_added = True
                    if isinstance(max_loss, torch.Tensor):
                        max_loss.backward()
                    
                    if hasattr(self, 'max_jumps') and self.max_jumps > 0 and len(supp_samples) >= self.max_jumps:
                        break_flag = True
        
            elif discrete and supp_loss_float <= 0:
                if max_loss <= 0:
                    updated_best_loss = supp_loss_float
                    updated_best_net = copy.deepcopy(learner)
                    if beta is not None:
                        updated_best_net = safe_set_beta(updated_best_net, beta)
                    break_flag = True
                else:
                    max_loss.backward()
                    if not max_jumps_reached:
                        supp_samples.add(sorted_keys[0])
                        new_supp_added = True
                        if hasattr(self, 'max_jumps') and self.max_jumps > 0 and len(supp_samples) >= self.max_jumps:
                            break_flag = True
                    else:
                        break_flag = True
            
            else:
                # Line 13: Subgradients of loss for samples in M
                new_supp = False
                if isinstance(supp_loss, torch.Tensor):
                    supp_loss.backward(retain_graph=True)
                    supp_grads = torch.hstack([torch.flatten(param.grad) if param.grad is not None else torch.tensor([]) for param in learner.parameters()])
                    
                    # Line 15: If there is a misaligned subgradient (inner ≤ 0)
                    for k in sorted_keys:
                        optimizer.zero_grad()
                        losses[k].backward(retain_graph=True)
                        grads = torch.hstack([torch.flatten(param.grad) if param.grad is not None else torch.tensor([]) for param in learner.parameters()])
                        inner = torch.inner(grads, supp_grads)
                        
                        if inner <= 0 and not max_jumps_reached:
                            # Line 19: C ← C ∪ {ξ̄}
                            supp_samples.add(k)
                            new_supp_added = True
                            if hasattr(self, 'max_jumps') and self.max_jumps > 0 and len(supp_samples) >= self.max_jumps:
                                break_flag = True
                            break
                    
                    if not new_supp_added:
                        optimizer.zero_grad()
                        supp_loss.backward()
        else:
            # Line 18: If no sample exceeds compression set, use max loss
            # Line 19: C ← C ∪ {ξ̄}
            if not max_jumps_reached:
                supp_samples.add(sorted_keys[0])
                new_supp_added = True
                if hasattr(self, 'max_jumps') and self.max_jumps > 0 and len(supp_samples) >= self.max_jumps:
                    break_flag = True
            else:
                break_flag = True
                
            optimizer.zero_grad()
            max_loss.backward()
        
        return break_flag, new_supp_added, supp_samples, updated_best_loss, updated_best_net

    def learn(
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: dict[str, torch.Tensor],
        Sdot: dict[str, torch.Tensor],
        Sind: Union[dict[str, list], None] = None,
        times: Union[dict[str, torch.Tensor], None] = None,
        best_loss: float = float('inf'),
        best_net: Union[learner.LearnerNN, None] = None,
        f_torch=None,
        discrete: bool = False,
        compression_set: Union[set, None] = None,
    ) -> dict:
        """
        Learns a certificate.

        Args:
            learner: fossil learner object (inherits from torch.nn.Module )
            optimizer: torch optimiser object
            S: dict of tensors of data (keys are domain names the data corresponds to, e.g. XD, XI)
            Sdot: dict of tensors containing f(data) (keys are domain names the data corresponds to, e.g. XD, XI)
            Sind: dict of indices for computing losses 
            times: list of timestamps for each trajectory
            best_loss: current best loss value
            best_net: current best network
            f_torch: torch function that computes f(data) (optional, for control synthesis)
            discrete: whether the system is discrete-time

        Returns:
            dict: empty dictionary



        This function is called by the learner object. It uses the sample Pytorch data points S and Sdot to
        calculate a loss function that should be minimised so the certificate properties are satisfied. The
        learn function does not return anything, but updates the optimiser object through the optimiser.step()
        function (which in turn updates the learners weights.)

        For control synthesis, the f_torch function is passed to the certificate, which is used to recompute the
        dynamics Sdot from the data S at each loop, since the control synthesis changes with each iteration.
        """
        raise NotImplementedError("Not implemented in " + self.__class__.__name__)

    def get_constraints(self, verifier, C, Cdot) -> tuple:
        """
        Returns (negation of) contraints for the certificate.
        The constraints are returned as a tuple of dictionaries, where each dictionary contains the constraints
        that should be verified together. For simplicity, as single dictionary may be returned, but it may be useful
        to verify the most difficult constraints last. If an earlier constraint is not satisfied, the later ones
        will not be checked.
        The dictionary keys are the domain names the constraints correspond to, e.g. XD, XI, XU.

        Logical operators are provided by the verifier object, e.g. _And, _Or, _Not, using the solver_fncts method,
        which returns a dictionary of functions. Eg. _And = verifier.solver_fncts()["And"].

        Example certificates assume that domains are in the form of SMT formulae, and that the certificate stores them
        as instance attributes from the __init__. User defined certificates may follow a different format, but should
        be consistent in how they are stored and accessed. They are passed to the certificate as a dictionary of
        domain names and symbolic domains as SMT formulae, this cannot be changed.

        Args:
            verifier: fossil verifier object
            C: SMT formula of Certificate
            Cdot: SMT formula of Certificate time derivative or one-step difference (for discrete systems)

        Returns:
            tuple: tuple of dictionaries of certificate conditons


        """
        raise NotImplementedError("Not implemented in " + self.__class__.__name__)

    @staticmethod
    def _assert_state(domains, data):
        """Checks that the domains and data are as expected for this certificate.

        This function is an optional debugging tool, but is called within CEGIS so should not be removed or
        renamed, and should only raise an exception if the domains or data are not as expected.
        """
        pass


class Practical_Lyapunov(Certificate):
    """
    Certificies stability for CT and DT models
    bool LLO: last layer of ones in network
    XD: Symbolic formula of domain
    XG: Goal region (around origin)
    """

    bias = False

    def __init__(self, domains, config: ScenAppConfig) -> None:
        self.domain = domains[XD]
        self.llo = config.LLO
        self.control = config.CTRLAYER is not None
        self.D = config.DOMAINS
        self.beta = None
        self.T = config.SYSTEM.time_horizon
    
    def compute_state_loss(
            self, 
            V_I: torch.Tensor, 
            V_G: torch.Tensor,
            V_D: torch.Tensor,
            V_SD: torch.Tensor,
            V_D_lie: torch.Tensor,
            beta: torch.Tensor,
            Vdot: torch.Tensor, 
            indices: dict,
            supp_samples: set,
    ) -> tuple[torch.Tensor, dict]:
        """_summary_

        Args:
            V (torch.Tensor): Lyapunov samples over domain
            Vdot (torch.Tensor): Lyapunov derivative samples over domain
            circle (torch.Tensor): Circle

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        
        relu = torch.nn.ReLU()
        
        init_loss = V_I
        border_loss = -V_SD
        goal_loss = V_G-(V_I.min()+V_D.min())/2#minus since V_I<0
        state_loss = -V_D+beta
        
        margin = 1e-5
        
        init_con = relu(init_loss+margin).mean()
        border_con = relu(border_loss+margin).mean()
        state_con = relu(state_loss+margin).mean()
        goal_con = relu(goal_loss+margin).mean()

        psi_s = state_con+border_con+init_con+goal_con
        return psi_s

    def compute_loss(
            self, 
            V_I: torch.Tensor, 
            V_G: torch.Tensor,
            V_D: torch.Tensor,
            V_SD: torch.Tensor,
            V_D_lie: torch.Tensor,
            beta: torch.Tensor,
            Vdot: torch.Tensor, 
            indices: dict,
            supp_samples: set,
    ) -> tuple[dict, Union[torch.Tensor, int], dict]:
        """_summary_

        Args:
            V (torch.Tensor): Lyapunov samples over domain
            Vdot (torch.Tensor): Lyapunov derivative samples over domain
            circle (torch.Tensor): Circle

        Returns:
            tuple[dict, Union[torch.Tensor, int], dict]: losses, supp_loss, and accuracy
        """

        
        relu = torch.nn.ReLU()
        
        init_loss = V_I
        border_loss = -V_SD
        goal_loss = V_G-V_I.min()
        #goal_loss = V_G-(V_I.min()+V_D.min())/2#minus since V_I<0 #this gives better convergence and epsilon for nonabsorbing, but prefer to have it only in the warm start since no theoretical basis for it...
        state_loss = -V_D+beta
        
        margin = 1e-5
        
        init_con = relu(init_loss+margin).mean()
        
        border_con = relu(border_loss+margin).mean()
        state_con = relu(state_loss+margin).mean()
        goal_con = relu(goal_loss+margin).mean()
        psi_s = state_con+border_con+init_con+goal_con
        req_diff = ((V_I.max()-beta)/self.T)

        # Code below for trying to get samples before V<beta
        Vdot_selected = []
        selected_inds = []
        curr_ind = 0
        for inds in indices["lie"]:
            try:
                final_ind = inds[0]+torch.where(V_D_lie[inds]<beta)[0][0] 
            except IndexError:
                final_ind = inds[-1]+1
            selected = range(inds[0],final_ind)
            selected_inds.append(range(curr_ind,curr_ind+len(selected))) 
            curr_ind += len(selected)
            Vdot_selected.append(Vdot[selected])
        Vdot_selected = torch.hstack(Vdot_selected)
        lie_loss = Vdot_selected+relu(req_diff)
        if psi_s != 0:
            lie_loss = relu(lie_loss)
        
        valid_Vdot = True
        if len(lie_loss) == 0:
            losses = {-1: psi_s}
            valid_Vdot = False
        

        if valid_Vdot:
            supp_max = torch.tensor([-1.])
            ind_lie_max = lie_loss.argmax()
            
            sub_sample = -1
            
            #for i, elem in enumerate(selected_inds):
            #    if ind_lie_max in elem:
            #        sub_sample = i
            #        break
            for ind in supp_samples:
                inds = selected_inds[ind]
                if len(inds) > 0:
                    supp_max = torch.max(supp_max, lie_loss[inds].max())
            supp_loss = supp_max
        
            lie_losses = {}
            if supp_loss != -1:
                for i, elem in enumerate(selected_inds):
                    elem_lie_loss = lie_loss[elem].max()
                    if elem_lie_loss >= supp_max:
                        lie_losses[i] = elem_lie_loss
                    #if ind_lie_max in elem:
                    #    sub_sample = i
                    #    break
                
                supp_loss = supp_max
                losses = {k: lie_losses[k]+psi_s for k in lie_losses} 
                if not losses:  # If lie_losses is empty, create a default entry
                    losses = {-1: psi_s}
            else:
                lie_max = lie_loss.max()
                ind_lie_max = lie_loss.argmax()
                
                losses = {-1: lie_max+psi_s}  # Default value
                for i, elem in enumerate(selected_inds):
                    if ind_lie_max in elem:
                        losses = {i: lie_max+psi_s}
                        break
        else:
            supp_loss = 0
            losses = {-1: psi_s}
        if supp_loss != -1:
            supp_loss = supp_loss + psi_s
        goal_accuracy = (V_G<V_I.min()).count_nonzero().item()/len(V_G)
        dom_accuracy = (V_D>beta).count_nonzero().item()/len(V_D)
        if len(Vdot_selected) > 0:
            lie_accuracy = (Vdot_selected <= -req_diff).count_nonzero().item()/len(Vdot_selected)
        else:
            lie_accuracy = 0
        accuracy = {"goal_acc": goal_accuracy * 100, "domain_acc" : dom_accuracy*100, "lie_acc" :lie_accuracy*100}

        return losses, supp_loss, accuracy

    def learn(
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: dict[str, torch.Tensor],
        Sdot: dict[str, torch.Tensor],
        Sind: Union[dict[str, list], None] = None,
        times: Union[dict[str, torch.Tensor], None] = None,
        best_loss: float = float('inf'),
        best_net: Union[learner.LearnerNN, None] = None,
        f_torch=None,
        discrete: bool = False,
        compression_set: Union[set, None] = None
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: dict of tensors of data
        :param Sdot: dict of tensors containing f(data)
        :param compression_set: Optional set to track compression set across calls
        :return: --
        """
        torch.set_num_threads(8)

        batch_size = len(S[XD])
        learn_loops = 1000
        samples = S[XD]
        
        i1 = S[XD].shape[0]
        idot1 = Sdot[XD].shape[0]
        
        i2 = S[XI].shape[0]
        idot2 = Sdot[XI].shape[0]

        i3 = S[XG_BORDER].shape[0]
        idot3 = len(Sdot[XG_BORDER])

        i4 = S[XG].shape[0]
        idot4 = len(Sdot[XG])

        idot5 = len(Sdot[XS_BORDER])

        samples = torch.cat([S[XD], S[XI], S[XG_BORDER],  S[XG], S[XS_BORDER]])

        samples_dot = Sdot[XD]

        samples_with_nexts = samples[:idot1]
        states_only = torch.cat([samples[idot1:i1], samples[i1+idot2:i1+i2], samples[i1+i2+idot3:i1+i2+i3], samples[i1+i2+i3+idot4:i1+i2+i3+i4], samples[i1+i2+i3+i4+idot5:]])
        assert times is not None, "times must be provided"
        time_tensor = times[XD]

        supp_samples = compression_set if compression_set is not None else set()
        
        state_sol = False
        best_supp_defd = False
        for t in range(learn_loops):
            optimizer.zero_grad()
            if self.control:
                if f_torch is None:
                    raise ValueError("f_torch must be provided when control synthesis is enabled")
                samples_dot = f_torch(samples)

            if state_sol:
                V1, Vdot, circle = learner.get_all(samples_with_nexts, samples_dot, time_tensor)
                V2 = learner(states_only)
                V = V2
                V_D = V[:i1-idot1]
                V_I = V[i1-idot1:i1+i2-idot1-idot2]
                V_SG = V[i1+i2-idot1-idot2:i1+i2+i3-idot1-idot2-idot3]
                V_G = V[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
                V_SD = V[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
                beta = V_SG.min()
                losses, supp_loss, learn_accuracy = self.compute_loss(V_I, V_G, V_D, V_SD, V1, beta, Vdot, Sind if Sind is not None else {}, supp_samples)
                
                sorted_keys = sorted(losses, key=lambda k: losses[k], reverse=True)
                max_loss = losses[sorted_keys[0]]

                if t % 100 == 0 or t == learn_loops - 1:
                    log_loss_acc(t, max_loss, learn_accuracy, learner.verbose)

                break_flag, _, supp_samples, best_loss, updated_best_net = self.subsurface_algorithm(
                    losses, supp_loss, supp_samples, best_loss, learner, optimizer, discrete, beta
                )
                
                if updated_best_net is not None:
                    best_net = updated_best_net
                
                if break_flag:
                    break
                    
                optimizer.step()
            else:
                state_itt = 0
                while True:
                    optimizer.zero_grad()
                    V1, Vdot, circle = learner.get_all(samples_with_nexts, samples_dot, time_tensor)
                    V2 = learner(states_only)
                    V = V2
                    V_D = V[:i1-idot1]
                    V_I = V[i1-idot1:i1+i2-idot1-idot2]
                    V_SG = V[i1+i2-idot1-idot2:i1+i2+i3-idot1-idot2-idot3]
                    V_G = V[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
                    V_SD = V[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
                    beta = V_SG.min()
                    state_itt += 1
                    loss = self.compute_state_loss(V_I, V_G, V_D, V_SD, V1, beta, Vdot, Sind if Sind is not None else {}, supp_samples)
                    if loss == 0:
                        state_sol=True
                        break
                    else:
                        if state_itt % 1000 == 0:
                            loss_val = loss[0] if isinstance(loss, tuple) else loss
                            loss_v = loss_val.item() if hasattr(loss_val, "item") else loss_val
                            cert_log.debug("{} - loss: {:.10f}".format(state_itt, loss_v))
                            
                        if isinstance(loss, torch.Tensor):
                            loss.backward()
                        optimizer.step()
                #loss = self.compute_state_loss(V_I, V_G, V_D, V_SD, V1, beta, Vdot, Sind, supp_samples, convex)
                #if loss == 0:
                #    state_sol = True
                #else:
                #    loss.backward()
                #    optimizer.step()
                
        if best_net is None:
            best_net = copy.deepcopy(learner)
            
        V1, Vdot, circle = best_net.get_all(samples_with_nexts, samples_dot, time_tensor)
        V2 = best_net(states_only)
        V = V2
        V_D = V[:i1-idot1]
        V_I = V[i1-idot1:i1+i2-idot1-idot2]
        V_SG = V[i1+i2-idot1-idot2:i1+i2+i3-idot1-idot2-idot3]
        V_G = V[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
        V_SD = V[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
        beta = V_SG.min()

        losses, supp_loss, learn_accuracy = self.compute_loss(V_I, V_G, V_D, V_SD, V1, beta, Vdot, Sind if Sind is not None else {}, supp_samples)

        max_k = max(losses, key=lambda k: losses[k])
        max_loss = losses[max_k]
        
        best_loss = losses[max_k]
        
        supp_samples = supp_samples.union(set([max_k]))
        best_net = safe_set_beta(best_net, beta)
        
        supp_samples.discard(-1)
        return {
            ScenAppStateKeys.loss: max_loss, 
            "best_loss": best_loss, 
            "best_net": best_net, 
            "compression_set": supp_samples,
            "compression_set_size": len(supp_samples)
        }

    def get_violations(self, certificate, certificate_dot, S, Sdot, times, state_data):
        req_diff = (certificate(state_data["init"]).max() - certificate(state_data["goal_border"]).min()) / self.T
        violated = 0
        true_violated = 0
        for i, (traj, traj_deriv, time) in enumerate(zip(S, Sdot, times)):
            traj, traj_deriv, time = torch.tensor(traj.T, dtype=torch.float32), torch.tensor(np.array(traj_deriv).T, dtype=torch.float32), torch.tensor(time, dtype=torch.float32)
            if self.D is None or self.D.get(XD) is None or self.D.get(XG) is None:
                continue
            valid_inds = torch.where(self.D[XD].check_containment(traj))
            traj = traj[valid_inds]
            traj_deriv = traj_deriv[valid_inds]
            time = time[valid_inds]
            pred_V = certificate(traj)
            pred_0 = certificate(torch.zeros_like(traj))
            pred_Vdot = certificate_dot(traj, traj_deriv, time)
            non_goal_inds = torch.where(domains.Complement(self.D[XG]).check_containment(traj))
            if len(non_goal_inds) == 0:
                true_violated += 1
            # We should check for value violations, but currently don't
            if any(pred_Vdot[non_goal_inds] > -req_diff):
                violated += 1
        return violated, true_violated

class BarrierAlt(Certificate):
    """
    Certifies Safety of a model using Lie derivative everywhere.

    Works for continuous and discrete models.

    The algorithm can be constrained with a maximum number of jumps (compression set size)
    by setting the MAX_JUMPS parameter in ScenAppConfig. This implements the a priori 
    jump limit algorithm where:
    1. Phase 1 is a sample-independent warm-start until state loss is minimized
    2. Phase 2 is the main loop with misaligned gradient checks
    3. The algorithm terminates when either convergence is reached or 
       the maximum number of jumps (MAX_JUMPS) is reached

    Use the USE_APRIORI_JUMPS flag to toggle between:
    - True: Use the modified algorithm with max jumps limit (default)
    - False: Use the vanilla algorithm without jump limits

    Arguments:
    domains {dict}: dictionary of string: domains pairs for a initial set, unsafe set and domain
    """

    def __init__(self, domains, config: ScenAppConfig) -> None:
        # Line 1: Function A(θ, D) -- θ is the parameter vector, D is the dataset
        self.domain = domains[XD]
        self.initial_s = domains[XI]
        self.unsafe_s = domains[XU]
        self.bias = True
        self.D = config.DOMAINS
        self.T=config.SYSTEM.time_horizon
        self.max_jumps = config.MAX_JUMPS
        self.use_apriori_jumps = config.USE_APRIORI_JUMPS

    def compute_state_loss(
        # Line 4: l^s(θ) > 0 -- sample-independent state loss
        self,
        B_i: torch.Tensor,
        B_u: torch.Tensor,
        B_d: torch.Tensor,
        Bdot_d: torch.Tensor,
        indices: Union[dict, None],
        supp_samples: set,
    ) -> torch.Tensor:
        relu = torch.nn.ReLU()
        
        unsafe_margin = 1e-5
        init_loss = (relu(B_i).mean())
        unsafe_loss = relu(-B_u+unsafe_margin).mean()
        psi_s = init_loss + unsafe_loss
        
        return psi_s


    def compute_loss(
        # Line 11: L(θ, ξ) -- sample-dependent loss for each sample ξ
        self,
        B_i: torch.Tensor,
        B_u: torch.Tensor,
        B_d: torch.Tensor,
        Bdot_d: torch.Tensor,
        indices: Union[dict, None],
        supp_samples: set,
    ) -> tuple[dict, Union[torch.Tensor, int], dict]:
        """Computes loss function for Barrier certificate.

        Also computes accuracy of the current model.

        Args:
            B_i (torch.Tensor): Barrier values for initial set
            B_u (torch.Tensor): Barrier values for unsafe set
            B_d (torch.Tensor): Barrier values for domain
            Bdot_d (torch.Tensor): Barrier derivative values for domain

        Returns:
            tuple[dict, Union[torch.Tensor, int], dict]: losses, supp_loss, and accuracy
        """
        torch.set_num_threads(8)
        # Line 6: g ← ∇_θ l^s(θ) -- gradient of state loss
        # Line 7: θ ← θ - αg -- step in direction of state loss gradient
        learn_accuracy = (B_i <= 0).count_nonzero().item() + (
            B_u > 0
        ).count_nonzero().item()
        percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(B_u) + len(B_i))
        relu = torch.nn.ReLU()

        req_diff = (B_u.min() - B_i.max())/self.T
        lie_loss = Bdot_d-req_diff
        lie_accuracy = (
            100 * ((Bdot_d < req_diff).count_nonzero()).item() / Bdot_d.shape[0]
        )
        
        unsafe_margin = 1e-5
        init_loss = (relu(B_i).mean())
        unsafe_loss = relu(-B_u+unsafe_margin).mean()
        psi_s = init_loss + unsafe_loss
        
        if psi_s > 0:
            lie_loss = relu(lie_loss)
        
        # Line 12: Find samples with loss greater than compression set loss
        supp_max = torch.tensor([-1.0])
        lie_losses = {}
        #lie_max = lie_loss.max() 
        
        ind_lie_max = lie_loss.argmax()
        
        if indices is None:
            lie_losses = {0: lie_loss.max()}
            losses = {0: lie_loss.max() + psi_s}
            supp_loss = -1
        else:
            for ind in supp_samples:
                lie_inds = indices["lie"][ind]
                if len(lie_inds) > 0:
                    if lie_loss[lie_inds].max() > supp_max:
                        supp_max = lie_loss[lie_inds].max()
                        supp_max_ind = ind
            
            #sub_sample = -1
            supp_loss = supp_max

            if supp_loss != -1:
                # Line 13: Subgradients of loss for samples in M
                for i, elem in enumerate(indices["lie"]):
                    elem_lie_loss = lie_loss[elem].max()
                    if elem_lie_loss >= supp_max:
                        lie_losses[i] = elem_lie_loss
                #if ind_lie_max in elem:
                #    sub_sample = i
                #    break
                
                supp_loss = supp_max
                # Line 16: losses = {k: lie_losses[k]+ψ_s for k in lie_losses}
                losses = {k: lie_losses[k]+psi_s for k in lie_losses} 
            else:
                # Line 18: If no sample exceeds compression set, use max loss
                lie_max = lie_loss.max()
                ind_lie_max = lie_loss.argmax()
                losses = {-1: lie_max+psi_s}  # Default value
                for i, elem in enumerate(indices["lie"]):
                    if ind_lie_max in elem:
                        losses = {i: lie_max+psi_s}
                        break

        #loss = loss + psi_s
        if supp_loss != -1:
            supp_loss = supp_loss + psi_s

        accuracy = {
            "acc init unsafe": percent_accuracy_init_unsafe,
            "acc lie": lie_accuracy,
        }
        return losses, supp_loss, accuracy
    
    def learn(
        # Algorithm 2, Line 1: Fix {ξ^i}_{i=1}^N
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: dict[str, torch.Tensor],
        Sdot: dict[str, torch.Tensor],
        Sind: Union[dict[str, list], None] = None,
        times: Union[dict[str, torch.Tensor], None] = None,
        best_loss: float = float('inf'),
        best_net: Union[learner.LearnerNN, None] = None,
        f_torch=None,
        discrete: bool = False,
        compression_set: Union[set, None] = None,
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: dict of tensors of data
        :param Sdot: dict of tensors containing f(data)
        :param compression_set: Optional set to track compression set across calls
        :return: --
        """
        # print(f"DEBUG: BarrierAlt.learn received compression_set={compression_set}, size={len(compression_set) if compression_set else 0}")

        learn_loops = 1000
        condition_old = False
        i1 = S[XD].shape[0]
        idot1 = Sdot[XD].shape[0]
        i2 = S[XI].shape[0]
        idot2 = Sdot[XI].shape[0]
        if type(Sdot[XU]) is not list:
            idot3 = Sdot[XU].shape[0]
        else:
            idot3 = 0
        label_order = [XD, XI, XU]
        samples = torch.cat([S[label] for label in label_order if type(S[label]) is not list])
        samples_with_nexts = torch.cat([samples[:idot1], samples[i1:i1+idot2], samples[i1+i2:i1+i2+idot3]])
        states_only = torch.cat([samples[idot1:i1], samples[i1+idot2:i1+i2], samples[i1+i2+idot3:]])
        assert times is not None, "times must be provided"
        time_tensor = torch.cat([times[label] for label in label_order if type(times[label]) is not list])
        samples_dot = torch.cat([Sdot[label] for label in label_order if type(Sdot[label]) is not list])
        
        # Line 3: C ← ∅ (Initialize compression set)
        supp_samples = compression_set if compression_set is not None else set()
        
        jumps_count = len(supp_samples)
        
        state_sol = False
        prev_supp_loss = -1000
        best_supp_defd = False
        for t in range(learn_loops):
            optimizer.zero_grad()

            # Line 4: While l^s(θ) > 0
            B, Bdot, _ = learner.get_all(samples_with_nexts, samples_dot, time_tensor)
            
            B2 = learner(states_only)
            (
                B_d,
                Bdot_d,
            ) = (
                    B2[:i1-idot1] ,
                Bdot[:idot1],
            )
            B_i = B2[i1-idot1:i1+i2-idot2-idot1]
            B_u = B2[i1+i2-idot1-idot2:]
            if state_sol:
                # Line 11: Compute sample-dependent loss for all samples
                losses, supp_loss, accuracy = self.compute_loss(B_i, B_u, B_d, Bdot_d, Sind, supp_samples)

                sorted_keys = sorted(losses, key=lambda k: losses[k], reverse=True)
                max_loss = losses[sorted_keys[0]]

                if (t % int(learn_loops / 10) == 0 or learn_loops - t < 10) or t == 1:
                    log_loss_acc(t, max_loss, accuracy, learner.verbose)
                
                break_flag, _, supp_samples, best_loss, updated_best_net = self.subsurface_algorithm(
                    losses, supp_loss, supp_samples, best_loss, learner, optimizer, discrete, None
                )
                
                if updated_best_net is not None:
                    best_net = updated_best_net
                
                if break_flag:
                    break

                supp_loss_float = supp_loss.item() if isinstance(supp_loss, torch.Tensor) else float(supp_loss)
                prev_supp_loss = supp_loss_float
                optimizer.step()
            else:
                # Lines 4-9: While l^s(θ) > 0, minimize state loss
                state_itt = 0
                while True:
                    optimizer.zero_grad()
                    B, Bdot, _ = learner.get_all(samples_with_nexts, samples_dot, time_tensor)
                    
                    B2 = learner(states_only)
                    (
                        B_d,
                        Bdot_d,
                    ) = (
                            B2[:i1-idot1] ,
                        Bdot[:idot1],
                    )
                    B_i = B2[i1-idot1:i1+i2-idot2-idot1]
                    B_u = B2[i1+i2-idot1-idot2:]
                    state_itt += 1
                    loss = self.compute_state_loss(B_i, B_u, B_d, Bdot_d, Sind, supp_samples)
                    if loss == 0:
                        state_sol=True
                        break
                    else:
                        if state_itt % 100 == 0:
                            loss_v = loss.item() if hasattr(loss, "item") else loss
                            cert_log.debug("{} - loss: {:.5f}".format(state_itt, loss_v))
                        loss.backward()
                        optimizer.step()

        # Line 25: After convergence, return θ, C_N = C ∪ argmax_{ξ∈D} L(θ, ξ)
        if best_net is None:
            best_net = copy.deepcopy(learner)
            
        B, Bdot, _ = best_net.get_all(samples_with_nexts, samples_dot, time_tensor)
        B2 = best_net(states_only)
        (
            B_d,
            Bdot_d,
        ) = (
                B2[:i1-idot1] ,
            Bdot[:idot1],
        )
        B_i = B2[i1-idot1:i1+i2-idot2-idot1]
        B_u = B2[i1+i2-idot1-idot2:]
        losses, supp_loss, accuracy = self.compute_loss(B_i, B_u, B_d, Bdot_d, Sind, supp_samples)

        max_k = max(losses, key=lambda k: losses[k])
        max_loss = losses[max_k]
        best_loss = losses[max_k]
        
        if not self.use_apriori_jumps or self.max_jumps == -1 or jumps_count < self.max_jumps:
            supp_samples = supp_samples.union(set([max_k]))
            jumps_count += 1
        
        supp_samples.discard(-1)
        
        return {
            ScenAppStateKeys.loss: max_loss,
            "best_loss": best_loss,
            "best_net": best_net,
            "compression_set": supp_samples,
            "compression_set_size": len(supp_samples),
            "jumps_count": jumps_count
        }

    def get_violations(self, certificate, certificate_dot, S, Sdot, times, state_data):
        if self.D is None or self.D.get(XD) is None or self.D.get(XI) is None or self.D.get(XU) is None:
            return 0, 0
        req_diff = (certificate(state_data["unsafe"]).min() - certificate(state_data["init"]).max()) / self.T
        true_violated = 0
        violated = 0
        for traj, traj_deriv, time in zip(S, Sdot, times):
            traj = torch.tensor(traj.T, dtype=torch.float32)
            traj_deriv = torch.tensor(np.array(traj_deriv).T, dtype=torch.float32)
            time = torch.tensor(time, dtype=torch.float32)
            valid_inds = torch.where(self.D[XD].check_containment(traj))
            traj = traj[valid_inds]
            traj_deriv = traj_deriv[valid_inds]
            time = time[valid_inds]
            initial_inds = torch.where(self.D[XI].check_containment(traj))
            unsafe_inds = torch.where(self.D[XU].check_containment(traj))
            pred_B_i = certificate(traj[initial_inds])
            pred_B_u = certificate(traj[unsafe_inds])
            pred_B_dots = certificate_dot(traj, traj_deriv, time)
            if any(self.D[XU].check_containment(traj)):
                true_violated += 1
            #if (any(pred_B_i >= 0) or
            #        any(pred_B_u <= 0)):
            #    raise ValueError("Value violation!")
            if any(pred_B_dots > req_diff):
                violated += 1
        return violated, true_violated



class RWS(Certificate):
    """Certificate to satisfy a reach-while-stay property.

    Reach While stay must satisfy:
    forall x in XI, V <= 0,
    forall x in boundary of XS, V > 0,
    forall x in A \\ XG, dV/dt < 0
    A = {x \\in XS| V <=0 }

    """

    def __init__(self, domains, config: ScenAppConfig) -> None:
        """initialise the RWS certificate
        Domains should contain:
            XI: compact initial set
            XS: compact safe set
            dXS: safe border
            XG: compact goal set
            XD: whole domain

        Data sets for learn should contain:
            SI: points from XI
            SU: points from XD \\ XS
            SD: points from XS \\ XG (domain less unsafe and goal set)

        """
        self.domain = domains[XD]
        self.initial = domains[XI]
        self.safe = domains[XS]
        self.safe_border = domains[XS_BORDER]
        self.goal = domains[XG]
        self.bias = True
        self.BORDERS = (XS,)
        self.D = config.DOMAINS
        self.T = config.SYSTEM.time_horizon

    def compute_state_loss(self, V_i, V_u, V_d, V_d_states, V_g, Vdot_d, beta, indices: Union[dict, None], supp_samples):
        margin = 1e-5
        margin_lie = 0.0
        acc_init = (V_i <= -margin).count_nonzero().item()*100/len(V_i)
        acc_unsafe = (V_u >= margin).count_nonzero().item()*100/len(V_u)
        acc_domain = (V_d_states > beta).count_nonzero().item()*100/len(V_d_states)
        relu = torch.nn.ReLU()

        Vdot_selected = []
        selected_inds = []
        curr_ind = 0
        
        init_loss = relu(V_i + margin).mean()
        unsafe_loss = relu(-V_u + margin).mean()
        state_loss = relu(-V_d_states + beta+margin).mean()
        goal_loss = relu(V_g-(V_i.min()+V_d_states.min())/2+margin).mean()
        
        psi_s = init_loss+unsafe_loss+state_loss+goal_loss
        accuracy = {
        "acc init": acc_init,
        "acc unsafe": acc_unsafe,
        "acc domain": acc_domain,
        }
        return psi_s, accuracy

    def compute_loss(self, V_i, V_u, V_d, V_d_states, V_g, Vdot_d, beta, indices: Union[dict, None], supp_samples) -> tuple[dict, Union[torch.Tensor, int], dict]:
        # V_d must match Vdot_d
        margin = 1e-5
        margin_lie = 0.0
        acc_init = (V_i <= -margin).count_nonzero().item()*100/len(V_i)
        acc_unsafe = (V_u >= margin).count_nonzero().item()*100/len(V_u)
        acc_domain = (V_d_states > beta).count_nonzero().item()*100/len(V_d_states)
        relu = torch.nn.ReLU()

        Vdot_selected = []
        selected_inds = []
        Vdot_unselected = []
        unselected_inds = []
        curr_ind = 0
        curr_un_ind = 0

        init_loss = relu(V_i + margin).mean()
        unsafe_loss = relu(-V_u + margin).mean()
        state_loss = relu(-V_d_states + beta+margin).mean()
        goal_loss = relu(V_g-V_i.min()+margin).mean()#minus since V_I<0
        
        psi_s = init_loss+unsafe_loss+state_loss+goal_loss
        
        if indices is None:
            supp_loss = 0 
            losses = {-1: psi_s}
            lie_accuracy = 0.0
            accuracy = {
                "acc init": acc_init,
                "acc unsafe": acc_unsafe,
                "acc domain": acc_domain,
                "acc lie": lie_accuracy,
            }
            return losses, supp_loss, accuracy
            
        for inds in indices["lie"]:
            try:
                final_ind = inds[0]+torch.where(V_d[inds]<beta)[0][0] 
            except IndexError:
                final_ind = inds[-1]+1
            selected = range(inds[0],final_ind)
            unselected = range(final_ind, inds[-1])
            selected_inds.append(range(curr_ind,curr_ind+len(selected))) 
            unselected_inds.append(range(curr_un_ind, curr_un_ind+len(unselected)))

            curr_ind += len(selected)
            curr_un_ind += len(unselected)

            Vdot_selected.append(Vdot_d[selected])
            
            Vdot_unselected.append(Vdot_d[unselected])

        Vdot_selected = torch.hstack(Vdot_selected)
        Vdot_unselected = torch.hstack(Vdot_unselected)

        req_diff = relu((V_i.max()-beta)/self.T)
        
        lie_loss = Vdot_selected+req_diff
        
        if psi_s > 0:
            lie_loss = relu(lie_loss)

        req_diff_2 = relu((V_u.min()-beta)/self.T)

        barr_lie_loss=Vdot_unselected-req_diff_2
        
        if psi_s > 0:
            barr_lie_loss = relu(barr_lie_loss)

        valid_Vdot = True
        lie_accuracy = 0.0  # Ensure lie_accuracy is always defined
        if len(lie_loss) == 0:
            loss = 0.0
            # plus 0.1 so this doesn't accidentally lead to loss = 0
            supp_loss = -1
            valid_Vdot = False

        if valid_Vdot:
            # this might need changing in case there are points in the unsafe or goal set?
            # ensure no goal states in domain data (see rwa_2 for example)
            if Vdot_selected.shape[0] > 0:
                lie_accuracy = (((Vdot_selected <= -req_diff).count_nonzero()).item() * 100 / Vdot_selected.shape[0])
            else:
                lie_accuracy = 0.0
            supp_max = torch.tensor([-1.0])
            losses = {}
            lie_losses = {}

            for ind in supp_samples:
                lie_sel_inds = selected_inds[ind]
                lie_unsel_inds = unselected_inds[ind]
                if len(lie_sel_inds) > 0:
                    supp_max = torch.max(supp_max, lie_loss[lie_sel_inds].max())
                #adjusted_inds = torch.cat([torch.where(lie_index[:,0] == elem)[0] for elem in lie_inds])
                if len(lie_unsel_inds) > 0:
                    supp_max = torch.max(supp_max, barr_lie_loss[lie_unsel_inds].max())
            supp_loss = supp_max 

            for i, (sel_inds, unsel_inds) in enumerate(zip(selected_inds, unselected_inds)):
                if supp_loss != -1:
                    if len(sel_inds) > 0:
                        elem_r_loss = lie_loss[sel_inds].max()
                    else:
                        elem_r_loss = torch.tensor([-1.0])
                    if len(unsel_inds) > 0:
                        elem_barr_loss = barr_lie_loss[unsel_inds].max()
                    else:
                        elem_barr_loss = torch.tensor([-1.0])

                    elem_lie_loss = torch.max(elem_r_loss, elem_barr_loss)
                    if elem_lie_loss >= supp_max:
                        lie_losses[i] = elem_lie_loss
                    losses = {k: lie_losses[k]+psi_s for k in lie_losses} 
                else:
                    lie_r_max = lie_loss.max()
                    lie_b_max = barr_lie_loss.max()
                    if lie_r_max > lie_b_max:
                        ind_lie_max = lie_loss.argmax()
                        lie_max = lie_r_max
                        losses = {-1: lie_max+psi_s}  # Default value
                        for i, elem in enumerate(selected_inds):
                            if ind_lie_max in elem:
                                losses = {i: lie_max+psi_s}
                                break
                    else:
                        ind_lie_max = barr_lie_loss.argmax()
                        lie_max = lie_b_max
                        losses = {-1: lie_max+psi_s}  # Default value
                        for i, elem in enumerate(unselected_inds):
                            if ind_lie_max in elem:
                                losses = {i: lie_max+psi_s}
                                break
        else:
            supp_loss = 0 
            losses = {-1: psi_s}
            lie_accuracy = 0.0
        if supp_loss != -1:
            supp_loss = supp_loss + psi_s

        accuracy = {
            "acc init": acc_init,
            "acc unsafe": acc_unsafe,
            "acc domain": acc_domain,
            "acc lie": lie_accuracy,
        }

        return losses, supp_loss, accuracy

    def learn(
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: dict[str, torch.Tensor],
        Sdot: dict[str, torch.Tensor],
        Sind: Union[dict[str, list], None] = None,
        times: Union[dict[str, torch.Tensor], None] = None,
        best_loss: float = float('inf'),
        best_net: Union[learner.LearnerNN, None] = None,
        f_torch = None,
        discrete: bool = False,
        compression_set: Union[set, None] = None,
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: dict of tensors of data
        :param Sdot: dict of tensors containing f(data)
        :param compression_set: Optional set to track compression set across calls
        :return: --
        """
        torch.set_num_threads(8)
        
        assert len(S) == len(Sdot)

        learn_loops = 1000
        condition_old = False
        i1 = S[XD].shape[0]
        idot1 = Sdot[XD].shape[0]
        i2 = S[XI].shape[0]
        idot2 = Sdot[XI].shape[0]
        
        i3 = S[XG].shape[0]
        idot3 = 0
        if type(Sdot[XS_BORDER]) is not list:
            idot4 = Sdot[XS_BORDER].shape[0]
        else:
            idot4 = 0
        i4 = S[XS_BORDER].shape[0]

        idot5 = len(Sdot[XG_BORDER])

        label_order = [XD, XI, XG, XS_BORDER, XG_BORDER]
        samples = torch.cat([S[label] for label in label_order if type(S[label]) is not list])

        samples_dot = Sdot[XD]
        samples_with_nexts = S[XD][:idot1]
        states_only = torch.cat([samples[idot1:i1], samples[i1+idot2:i1+i2], samples[i1+i2+idot3:i1+i2+i3], samples[i1+i2+i3+idot4:i1+i2+i3+i4], samples[i1+i2+i3+i4+idot5:]])
        if times is not None and not isinstance(times, dict):
            raise TypeError("times must be a dict[str, torch.Tensor] or None")
        times_cat = torch.cat([times[label] for label in label_order if type(times[label]) is not list]) if times is not None else None
        
        supp_samples = compression_set if compression_set is not None else set()
        
        state_sol = False
        best_supp_defd = False
        for t in range(learn_loops):
            optimizer.zero_grad()

            B_d, Bdot_d, _ = learner.get_all(samples_with_nexts, samples_dot, times_cat[:idot1] if times_cat is not None else None)

            B = learner(states_only)
            
            B_d_states = B[:i1-idot1]
            B_i = B[i1-idot1 : i1 + i2-idot1-idot2]
            B_g = B[i1 + i2-idot1-idot2 :i1+i2+i3-idot1-idot2-idot3]
            B_u = B[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
            B_sg = B[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
            beta = B_sg.min()
            
            if state_sol:
                losses, supp_loss, accuracy = self.compute_loss(B_i, B_u, B_d, B_d_states, B_g, Bdot_d, beta, Sind, supp_samples)
                sorted_keys = sorted(losses, key=lambda k: losses[k], reverse=True)
                max_loss = losses[sorted_keys[0]]
                if (t-1) % int(learn_loops / 100) == 0 or learn_loops - t < 10:
                    log_loss_acc(t, max_loss, accuracy, learner.verbose)
                
                sorted_keys = sorted(losses, key=lambda k: losses[k], reverse=True)
                max_loss = losses[sorted_keys[0]]
                break_flag = False
                
                if supp_loss != -1:
                    supp_loss_float = supp_loss.item() if isinstance(supp_loss, torch.Tensor) else float(supp_loss)
                    if supp_loss_float < best_loss:
                        best_loss = supp_loss_float
                        best_net = copy.deepcopy(learner)
                        best_net = safe_set_beta(best_net, beta)
                    
                    optimizer.zero_grad()
                    prev_supp_loss = supp_loss_float
                    
                    # If (supp_loss - best_loss) >= η, add new sample to C
                    if (supp_loss_float - best_loss) >= 1e-1:
                        if sorted_keys[0] in supp_samples:
                            break_flag = True
                        else:
                            # C ← C ∪ {ξ̄}
                            supp_samples.add(sorted_keys[0])
                            if isinstance(max_loss, torch.Tensor):
                                max_loss.backward()
                    elif discrete and supp_loss_float <= 0:
                        if max_loss <= 0:
                            best_loss = supp_loss_float
                            best_net = copy.deepcopy(learner)
                            best_net = safe_set_beta(best_net, beta)
                            break_flag = True
                        else:
                            max_loss.backward()
                            supp_samples.add(sorted_keys[0])
                    else:
                        # Subgradients of loss for samples in M
                        new_supp = False
                        if isinstance(supp_loss, torch.Tensor):
                            supp_loss.backward(retain_graph=True)
                            supp_grads = torch.hstack([torch.flatten(param.grad) if param.grad is not None else torch.tensor([]) for param in learner.parameters()])
                            
                            # If there is a misaligned subgradient (inner ≤ 0)
                            for k in sorted_keys:
                                optimizer.zero_grad()
                                losses[k].backward(retain_graph=True)
                                grad = torch.hstack([torch.flatten(param.grad) if param.grad is not None else torch.tensor([]) for param in learner.parameters()])
                                inner = torch.inner(grad, supp_grads)
                                
                                if inner <= 0:
                                    # C ← C ∪ {ξ̄}
                                    supp_samples.add(k)
                                    new_supp = True
                                    break
                            
                            if not new_supp:
                                optimizer.zero_grad()
                                supp_loss.backward()
                else:
                    # If no sample exceeds compression set, use max loss
                    # C ← C ∪ {ξ̄}
                    supp_samples.add(sorted_keys[0])
                    if isinstance(max_loss, torch.Tensor):
                        max_loss.backward()
                
                if break_flag:
                    break
                
                optimizer.step()
            else:
                state_itt = 0
                while True:
                    optimizer.zero_grad()
                    B_d, Bdot_d, _ = learner.get_all(samples_with_nexts, samples_dot, times_cat[:idot1] if times_cat is not None else None)
                    B = learner(states_only)
                    B_d_states = B[:i1-idot1]
                    B_i = B[i1-idot1 : i1 + i2-idot1-idot2]
                    B_g = B[i1 + i2-idot1-idot2 :i1+i2+i3-idot1-idot2-idot3]
                    B_u = B[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
                    B_sg = B[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
                    beta = B_sg.min()
                    state_itt += 1
                    state_loss, accuracy = self.compute_state_loss(B_i, B_u, B_d, B_d_states, B_g, Bdot_d, beta, Sind, supp_samples)
                    if state_loss == 0:
                        state_sol = True
                        break
                    else:
                        state_loss.backward()
                        optimizer.step()
                        if state_itt % 100 == 0:
                            loss_v = state_loss.item() if hasattr(state_loss, "item") else state_loss
                            cert_log.debug("{} - loss: {:.5f}".format(state_itt, loss_v))
        if best_net is None:
            best_net = copy.deepcopy(learner)
        B_d, Bdot_d, _ = best_net.get_all(samples_with_nexts, samples_dot, times_cat[:idot1] if times_cat is not None else None)
        B = best_net(states_only)
        B_d_states = B[:i1-idot1]
        B_i = B[i1-idot1:i1+i2-idot2-idot1]
        B_g = B[i1 + i2-idot1-idot2 :i1+i2+i3-idot1-idot2-idot3]
        B_u = B[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
        B_sg = B[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
        beta = B_sg.min()
        losses, supp_loss, accuracy = self.compute_loss(B_i, B_u, B_d, B_d_states, B_g, Bdot_d, beta, Sind, supp_samples)
        max_k = max(losses, key=lambda k: losses[k])
        max_loss = losses[max_k]
        best_loss = losses[max_k]
        supp_samples = supp_samples.union(set([max_k]))
        best_net = safe_set_beta(best_net, beta)
        return {
            ScenAppStateKeys.loss: max_loss, 
            "best_loss": best_loss, 
            "best_net": best_net, 
            "compression_set": supp_samples,
            "compression_set_size": len(supp_samples)
        }

    def get_violations(self, certificate, certificate_dot, S, Sdot, times, state_data):
        if self.D is None or self.D.get(XI) is None or self.D.get(XG) is None or self.D.get(XS) is None:
            return 0, 0
        req_diff = (certificate(state_data["init"]).max() - certificate(state_data["goal"]).min()) / self.T
        true_violated = 0
        violated = 0
        for traj, traj_deriv, time in zip(S, Sdot, times):
            traj = torch.tensor(traj.T, dtype=torch.float32)
            traj_deriv = torch.tensor(np.array(traj_deriv).T, dtype=torch.float32)
            time = torch.tensor(time, dtype=torch.float32)
            if self.D is None or self.D.get(XI) is None or self.D.get(XG) is None or self.D.get(XS) is None:
                continue
            initial_inds = torch.where(self.D[XI].check_containment(traj))
            goal_inds = torch.where(self.D[XG].check_containment(traj))
            V_d = certificate(traj)
            pred_dots = certificate_dot(traj, traj_deriv, time)
            goal_inds_0 = torch.where(self.D[XG].check_containment(traj))[0]
            if not all(self.D[XS].check_containment(traj)) or not any(self.D[XG].check_containment(traj)):
                true_violated += 1
            lie_inds = torch.nonzero(V_d <= 0)
            if any(self.D[XG].check_containment(traj)):
                lie_inds = [ind.item() for ind in lie_inds if ind not in goal_inds_0]
            if any(pred_dots[lie_inds] > req_diff):
                violated += 1
        return violated, true_violated


    @staticmethod
    def _assert_state(domains, data):
        domain_labels = set(domains.keys())
        data_labels = set(data.keys())
        _set_assertion(
            set([XD, XI, XS, XS_BORDER, XG]), domain_labels, "Symbolic Domains"
        )
        _set_assertion(set([XD, XI, XU]), data_labels, "Data Sets")


class RSWS(RWS):
    """Reach and Stay While Stay Certificate

    Firstly satisfies reach while stay conditions, given by:
        forall x in XI, V <= 0,
        forall x in boundary of XS, V > 0,
        forall x in A \\ XG, dV/dt < 0
        A = {x \\in XS| V <=0 }

    http://arxiv.org/abs/1812.02711
    In addition to the RWS properties, to satisfy RSWS:
    forall x in border XG: V > \\beta
    forall x in XG \\ int(B): dV/dt <= 0
    B = {x in XS | V <= \\beta}
    Best to ask SMT solver if a beta exists such that the above holds -
    but currently we don't train for this condition.

    Crucially, this relies only on the border of the safe set,
    rather than the safe set itself.
    Since the border must be positive (and the safe invariant negative), this is inline
    with making the complement (the unsafe set) positive. Therefore, the implementation
    requires an unsafe set to be passed in, and assumes its border is the same of the border of the safe set.
    """

    def __init__(self, domains, config: ScenAppConfig) -> None:
        """initialise the RSWS certificate
        Domains should contain:
            XI: compact initial set
            XS: compact safe set
            dXS: safe border
            XG: compact goal set
            XD: whole domain

        Data sets for learn should contain:
            SI: points from XI
            SU: points from XD \\ XS
            SD: points from XS \\ XG (domain less unsafe and goal set)

        """
        raise NotImplementedError #This class currently not implemented
        self.domain = domains[XD]
        self.initial = domains[XI]
        self.safe = domains[XS]
        self.safe_border = domains[XS_BORDER]
        self.goal = domains[XG]
        self.goal_border = domains[XG_BORDER]
        self.BORDERS = (XS, XG)
        self.bias = True
        self.D = config.DOMAINS
        self.T = config.SYSTEM.time_horizon

    def compute_beta_loss(self, beta, V_g_border_min, V_g, Vdot_g, V_d, indices: Union[dict, None], supp_samples):
        """Compute the loss for the beta condition
        :param beta: the guess value of beta based on the min of V of XG_border
        :param V_d: the value of V at points in the goal set
        :param Vdot_d: the value of the lie derivative of V at points in the goal set"""
        lie_index = torch.nonzero(V_g <= V_g_border_min)
        
        relu = torch.nn.ReLU()

        req_diff = relu(V_g_border_min-beta)/self.T
        if indices is None:
            beta_loss = torch.Tensor([0])
            supp_beta_loss = -1 
            new_sub_samples = set()
            return beta_loss, supp_beta_loss, new_sub_samples
            
        if lie_index.nelement() != 0:
            beta_lie = relu(torch.index_select(Vdot_g, dim=0, index=lie_index[:, 0])-req_diff)
            accuracy = (beta_lie <= 0).count_nonzero().item() * 100 / beta_lie.shape[0]
            supp_max = torch.tensor([-1.0])
            lie_max = beta_lie.max()
            ind_lie_max = beta_lie.argmax()
            beta_loss = lie_max
            sub_sample = -1
            for i, elem in enumerate(indices["lie"]):
                if lie_index[ind_lie_max, 0] in elem:
                    sub_sample = i
                    break
            for ind in supp_samples:
                lie_inds = indices["lie"][ind]
                adjusted_inds = torch.cat([torch.where(lie_index[:,0] == elem)[0] for elem in lie_inds])
                if len(adjusted_inds) > 0:
                    supp_max = torch.max(supp_max, beta_lie[adjusted_inds].max())
            supp_beta_loss = supp_max
            new_sub_samples = set([sub_sample])
        else:
            # Do we penalise V > beta in safe set, or  V < beta in goal set?
            beta_loss = torch.Tensor([0])
            supp_beta_loss = -1 
            new_sub_samples = set()

        return beta_loss, supp_beta_loss, new_sub_samples

    def learn(
        self,
        learner: 'learner.LearnerNN',
        optimizer: Optimizer,
        S: dict[str, torch.Tensor],
        Sdot: dict[str, torch.Tensor],
        Sind: Union[dict[str, list], None] = None,
        times: Union[dict[str, torch.Tensor], None] = None,
        best_loss: float = float('inf'),
        best_net: Union['learner.LearnerNN', None] = None,
        f_torch=None,
        discrete: bool = False,
        compression_set: Union[set, None] = None,
        *,  # Keyword-only arguments after this
        convex: bool = True,
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: dict of tensors of data
        :param Sdot: dict of tensors containing f(data)
        :param compression_set: Optional set to track compression set across calls
        :param convex: Whether to use convex optimization approach
        :return: --
        """
        assert len(S) == len(Sdot)

        learn_loops = 1000
        
        lie_indices = Sdot[XD].shape[0], S[XD].shape[0]
        lie_dot_indices = 0, Sdot[XD].shape[0]

        init_indices = lie_indices[1]+Sdot[XI].shape[0], lie_indices[1] + S[XI].shape[0]

        unsafe_indices = init_indices[1]+len(Sdot[XU]), init_indices[1] + S[XU].shape[0]

        goal_border_indices = (
            unsafe_indices[1] + len(Sdot[XG_BORDER]),
            unsafe_indices[1] + S[XG_BORDER].shape[0],
        )

        goal_indices = (
            goal_border_indices[1]+len(Sdot[XG]),
            goal_border_indices[1] + S[XG].shape[0],
        )

        goal_dot_indices = goal_border_indices[1], goal_indices[0]
        # Setting label order allows datasets to be passed in any order
        label_order = [XD, XI, XU, XG_BORDER, XG]
        lie_label_order = [XD, XG] # make sure no goal data in XD
        samples = torch.cat([S[label] for label in label_order if type(S[label]) is not list])

        samples_dot = torch.cat([Sdot[label] for label in lie_label_order])
        
        samples_with_nexts = torch.cat([samples[:lie_dot_indices[1]], samples[goal_dot_indices[0]:goal_dot_indices[1]]])
        
        if times is not None and not isinstance(times, dict):
            raise TypeError("times must be a dict[str, torch.Tensor] or None")
        times_cat = torch.cat([times[label] for label in label_order if type(times[label]) is not list]) if times is not None else None
        
        supp_samples = compression_set if compression_set is not None else set()
        
        for t in range(learn_loops):
            optimizer.zero_grad()

            if times_cat is not None:
                times_for_get_all = torch.cat([times_cat[:lie_dot_indices[1]], times_cat[-len(Sdot[XG]):]])
            else:
                times_for_get_all = torch.tensor([])  # or appropriate empty tensor

            V, Vdot, _ = learner.get_all(samples_with_nexts, samples_dot, times_for_get_all)

            (
                V_d,
                gradV_d,
            ) = (V[: lie_dot_indices[1]], Vdot[: lie_dot_indices[1]])

            Vstates = learner(samples)

            V_d_states = Vstates[lie_indices[0]:lie_indices[1]]
            V_i = Vstates[init_indices[0]: init_indices[1]]
            V_u = Vstates[unsafe_indices[0]: unsafe_indices[1]]
            S_dg = samples[goal_border_indices[0]: goal_border_indices[1]]
            V_g = Vstates[goal_indices[0]: goal_indices[1]]

            Vdot_g = Vdot[lie_dot_indices[1]:]
            samples_dot_d = samples_dot[: lie_indices[1]]

            beta = learner.compute_minimum(S_dg)[0]+V_g.min()/100
            loss, supp_loss, accuracy, sub_sample  = self.compute_loss(V_i, V_u, V_d, V_d_states, V_g, gradV_d, beta, Sind, supp_samples, convex) #type: ignore

            beta2 = learner.compute_minimum(S_dg)[0]
            #beta_loss, supp_beta_loss, beta_sub_sample = 0, -1, set()
            # converges without beta loss
            beta_loss, supp_beta_loss, beta_sub_sample = self.compute_beta_loss(beta, beta2, V_g, Vdot_g, V_d_states, Sind, supp_samples, convex) # type: ignore
            loss = loss + beta_loss
            #loss = torch.max(loss,beta_loss)
            if supp_loss != -1:
                if supp_beta_loss != -1:   
                    supp_loss = supp_loss + supp_beta_loss
                    sub_sample = sub_sample.union(beta_sub_sample)
            else:
                supp_loss = supp_beta_loss
                sub_sample = beta_sub_sample

            if loss <= best_loss:
                best_loss = loss
                best_net = copy.deepcopy(learner)
                best_net = safe_set_beta(best_net, beta)

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                log_loss_acc(t, loss, accuracy, learner.verbose)

            if convex:
                loss.backward()
            else:
                loss.backward(retain_graph=True)
                grads = torch.hstack([torch.flatten(param.grad) if param.grad is not None else torch.tensor([]) for param in learner.parameters()])
                if supp_loss != -1:
                    optimizer.zero_grad()
                    supp_loss.backward(retain_graph=True)
                    supp_grads = torch.hstack([torch.flatten(param.grad) if param.grad is not None else torch.tensor([]) for param in learner.parameters()])
                    inner = torch.inner(grads, supp_grads)
                    if inner <= 0:
                        supp_samples = supp_samples.union(sub_sample)
                        optimizer.zero_grad()
                        loss.backward()
                else:
                    supp_samples = supp_samples.union(sub_sample)
            optimizer.step()
        if times_cat is not None:
            times_for_get_all = torch.cat([times_cat[:lie_dot_indices[1]], times_cat[-len(Sdot[XG]):]])
        else:
            times_for_get_all = torch.tensor([])  # or appropriate empty tensor
        V, Vdot, _ = learner.get_all(samples_with_nexts, samples_dot, times_for_get_all)
        (
            V_d,
            gradV_d,
        ) = (V[: lie_dot_indices[1]], Vdot[: lie_dot_indices[1]])


        Vstates = learner(samples)
        
        V_d_states = Vstates[lie_indices[0] : lie_indices[1]]
        V_i = Vstates[init_indices[0] : init_indices[1]]
        V_u = Vstates[unsafe_indices[0] : unsafe_indices[1]]
        S_dg = samples[goal_border_indices[0] : goal_border_indices[1]]
        V_g = Vstates[goal_indices[0] : goal_indices[1]]
        
        Vdot_g = Vdot[lie_dot_indices[1] :]
        samples_dot_d = samples_dot[: lie_indices[1]]
        
        beta = learner.compute_minimum(S_dg)[0]+V_g.min()/100
        
        loss, supp_loss, accuracy, sub_sample  = self.compute_loss(V_i, V_u, V_d, V_d_states, V_g, gradV_d, beta, Sind, supp_samples, convex) # type: ignore

        beta2 = learner.compute_minimum(S_dg)[0]
        beta_loss, supp_beta_loss, beta_sub_sample = self.compute_beta_loss(beta, beta2, V_g, Vdot_g, V_d_states, Sind, supp_samples, convex) # type: ignore
        #loss = torch.max(loss, beta_loss)
        loss = loss + beta_loss

        if loss <= best_loss:
            best_loss = loss
            best_net = copy.deepcopy(learner)
            best_net = safe_set_beta(best_net, beta)

        return {ScenAppStateKeys.loss: loss, "best_loss":best_loss, "best_net":best_net, "new_supps":supp_samples}


    def beta_search(self, learner, verifier, C, Cdot, S):
        return learner.compute_minimum(S[XG_BORDER])[0]
    
    def get_violations(self, certificate, certificate_dot, S, Sdot, times, state_data):
        req_diff = (certificate(state_data["init"]).max() - certificate(state_data["goal"]).min()) / self.T
        true_violated = 0
        violated = 0
        for traj, traj_deriv, time in zip(S, Sdot, times):
            traj = torch.tensor(traj.T, dtype=torch.float32)
            traj_deriv = torch.tensor(np.array(traj_deriv).T, dtype=torch.float32)
            time = torch.tensor(time, dtype=torch.float32)
            if self.D is None or self.D.get(XI) is None or self.D.get(XG) is None or self.D.get(XS) is None:
                continue

            #valid_inds = torch.where(self.D[XD].check_containment(traj))
            #
            #traj = traj[valid_inds]
            #traj_deriv = traj_deriv[valid_inds]
            #time = time[valid_inds]

            initial_inds = torch.where(self.D[XI].check_containment(traj))

            # getting too many violations, need to investigate

            goal_inds = torch.where(self.D[XG].check_containment(traj))
            V_d = certificate(traj)
            pred_dots = certificate_dot(traj, traj_deriv, time)
            goal_inds_0 = torch.where(self.D[XG].check_containment(traj))[0]
            if len(goal_inds_0) == 0:
                continue
            first_goal_ind = goal_inds_0[0]
            if not all(self.D[XS].check_containment(traj)) or not all(self.D[XG].check_containment(traj[first_goal_ind:])):
                true_violated += 1
            lie_inds = torch.nonzero(V_d <= 0)
            if any(self.D[XG].check_containment(traj)):
                lie_inds = [ind.item() for ind in lie_inds if ind not in goal_inds_0]
            if any(pred_dots[lie_inds] > req_diff):
                violated += 1
                continue
        return violated, true_violated
            


class DoubleCertificate(Certificate):
    """In Devel class for synthesising any two certificates together"""

    def __init__(self, domains, config: ScenAppConfig):
        self.certificate1 = None
        self.certificate2 = None

    def compute_loss(self, C1, C2, Cdot1, Cdot2):
        if self.certificate1 is None or self.certificate2 is None:
            raise AttributeError("certificate1 and certificate2 must be set before calling compute_loss.")
        loss1 = self.certificate1.compute_loss(C1, Cdot1)
        loss2 = self.certificate2.compute_loss(C2, Cdot2)
        return loss1[0] + loss2[0]

    def learn(
        self,
        learner: 'learner.LearnerNN',
        optimizer: Optimizer,
        S: dict[str, Any],
        Sdot: dict[str, Any],
        Sind: Union[dict[str, list], None] = None,
        times: Union[dict[str, Any], None] = None,
        best_loss: float = float('inf'),
        best_net: Union['learner.LearnerNN', None] = None,
        f_torch=None,
        discrete: bool = False,
        compression_set: Union[set, None] = None,
    ) -> dict:
        # Not implemented for DoubleCertificate
        raise NotImplementedError("DoubleCertificate.learn is not implemented.")

    def get_constraints(self, verifier, C, Cdot) -> tuple:
        """
        :param verifier: verifier object
        :param C: tuple containing SMT formula of Lyapunov function and barrier function
        :param Cdot: tuple containing SMT formula of Lyapunov lie derivative and barrier lie derivative
        :return: tuple of constraints from both certificates
        """
        C1, C2 = C
        Cdot1, Cdot2 = Cdot
        if self.certificate1 is None or self.certificate2 is None:
            raise AttributeError("certificate1 and certificate2 must be set before calling get_constraints.")
        cert1_cs = self.certificate1.get_constraints(verifier, C1, Cdot1)
        cert2_cs = self.certificate2.get_constraints(verifier, C2, Cdot2)
        # Ensure both are tuples (as required by base class)
        if not isinstance(cert1_cs, tuple):
            cert1_cs = tuple(cert1_cs)
        if not isinstance(cert2_cs, tuple):
            cert2_cs = tuple(cert2_cs)
        return cert1_cs + cert2_cs


class AutoSets:
    """Class for automatically handing sets for certificates"""

    def __init__(self, XD, certificate: CertificateType) -> None:
        self.XD = XD
        self.certificate = certificate
        self.sets = None  # Ensure self.sets is always defined

    def auto(self) -> tuple[dict, dict]:
        if self.certificate == CertificateType.PRACTICALLYAPUNOV:
            return self.auto_practical_lyap()
        elif self.certificate == CertificateType.SEQUENTIALREACH:
            return self.auto_sequential_reach()
        elif self.certificate == CertificateType.BARRIERALT:
            return self.auto_barrier_alt(self.sets)
        elif self.certificate == CertificateType.RWS:
            return self.auto_rws(self.sets)
        elif self.certificate == CertificateType.RSWS:
            return self.auto_rsws(self.sets)
        elif self.certificate == CertificateType.RAR:
            return self.auto_rar(self.sets)
        else:
            raise NotImplementedError("Unknown certificate type for auto()")

    def auto_lyap(self) -> tuple[dict, dict]:
        domains = {XD: self.XD}
        data = {XD: self.XD._generate_data(1000)}
        return domains, data

    def auto_practical_lyap(self) -> tuple[dict, dict]:
        raise NotImplementedError("auto_practical_lyap is not implemented.")

    def auto_sequential_reach(self, sets=None) -> tuple[dict, dict]:
        raise NotImplementedError("auto_sequential_reach is not implemented.")

    def auto_barrier_alt(self, sets=None) -> tuple[dict, dict]:
        raise NotImplementedError("auto_barrier_alt is not implemented.")

    def auto_rws(self, sets=None) -> tuple[dict, dict]:
        raise NotImplementedError("auto_rws is not implemented.")

    def auto_rsws(self, sets=None) -> tuple[dict, dict]:
        raise NotImplementedError("auto_rsws is not implemented.")

    def auto_rar(self, sets=None) -> tuple[dict, dict]:
        raise NotImplementedError("auto_rar is not implemented.")


def get_certificate(
    certificate: CertificateType, custom_cert=None
) -> Type[Certificate]:
    if certificate == CertificateType.PRACTICALLYAPUNOV:
        return Practical_Lyapunov
    # elif certificate == CertificateType.SEQUENTIALREACH:
    #     return Sequential_Reach
    elif certificate == CertificateType.BARRIERALT:
        return BarrierAlt
    elif certificate in (CertificateType.RWS, CertificateType.RWA):
        return RWS
    elif certificate in (CertificateType.RSWS, CertificateType.RSWA):
        return RSWS
    # elif certificate == CertificateType.RAR:
    #     return ReachAvoidRemain
    elif certificate == CertificateType.CUSTOM:
        if custom_cert is None:
            raise ValueError(
                "Custom certificate not provided (use ScenAppConfig CUSTOM_CERTIFICATE)))"
            )
        return custom_cert
    else:
        raise ValueError("Unknown certificate type {}".format(certificate))

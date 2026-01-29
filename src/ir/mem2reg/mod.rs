pub(crate) mod mem2reg;
pub(crate) mod phi_eliminate;

pub(crate) use mem2reg::promote_allocas;
pub(crate) use mem2reg::promote_allocas_with_params;
pub(crate) use phi_eliminate::eliminate_phis;

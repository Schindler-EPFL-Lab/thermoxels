from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax_fem.problem import Problem
from scipy.constants import convert_temperature


@dataclass(kw_only=True)
class TransientThermalProblem(Problem):
    """
    Time-dependent heat transfer problem using the FEM (Finite Element Method)
    """

    dt: float = 0.001
    """Delta time of the problem simulation."""
    simulation_time: float = 100.0
    """Total simualtion time."""
    num_steps: int = 20
    """Number of steps for the simulation"""
    h_natural: float = 8.0
    """Natural convection factor. Estimated value, typically between 5 and 10."""
    wind_speed: float = 5.56
    """Wind speed in [m/s]. Default value is 5.56 (20 km/h)."""
    solar_irradiance: float = 1000.0
    """Approximate solar irradiance in [W/m²]. Default value for a sunny day
    (https://en.wikipedia.org/wiki/Solar_irradiance)"""
    SB_constant: float = 5.67e-8
    """Stefan-Boltzmann constant."""
    emissivity: float = 0.85
    """ Emissivity of the material.
    (https://www.engineeringtoolbox.com/emissivity-coefficients-d_447.html)."""
    absorptivity: float = 0.6
    """Absorptivity of the material. Default value for concrete
    (https://www.engineeringtoolbox.com/radiation-surface-absorptivity-d_1805.html)"""
    rho: float = 2200.0
    """Material density [kg/m³]. Default value is for concrete
    (https://amesweb.info/Materials/Density-Materials.aspx)."""
    Cp: float = 880.0
    """Specific heat capacity [J/kg.K]. Default value is for concrete
    (https://www.engineeringtoolbox.com/specific-heat-capacity-d_391.html)."""
    k: float = 1.7
    """Thermal conductivity [W/m.K]."""
    exterior_temperature: float = convert_temperature(0, "Celsius", "Kelvin")
    """Exterior temperature [K]. Default value is 0°C, converted to Kelvin."""
    interior_temperature: float = convert_temperature(35, "Celsius", "Kelvin")
    """Interior temperature [K]. Default value is 35°C, converted to Kelvin."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.h_forced = TransientThermalProblem.wind_to_h(self.wind_speed)

    def get_mass_map(
        self,
    ) -> Callable[
        [
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
        jnp.ndarray,
    ]:
        """
        Returns a function to compute the transient term of the heat
        equation, representing the evolution of temperature in the material over time.

        :returns: Function to compute the transient term.
        """

        def transient_term(
            current_temperature: jnp.ndarray,
            point: jnp.ndarray,
            old_temperature: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Computes the transient term based on `current_temperature` and
            `old_temperature`.

            :returns: Transient heat term.
            """
            return (
                self.rho * self.Cp * (current_temperature - old_temperature) / self.dt
            )

        return transient_term

    def get_tensor_map(
        self,
    ) -> Callable[
        [
            jnp.ndarray,
            jnp.ndarray,
        ],
        jnp.ndarray,
    ]:
        """
        Returns a function to compute the heat conduction (diffusion) through the
        material term, based on Fourier's law.

        :returns: Function that computes the conduction term.
        """

        def heat_conduction_term(
            temperature_gradient: jnp.ndarray,
            old_temperature: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            The conduction term is proportional to the `temperature_gradient` and `k`.

            :returns: Heat conduction term.
            """
            conduction = self.k * temperature_gradient
            return conduction

        return heat_conduction_term

    def get_surface_maps(
        self,
    ) -> tuple[
        Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ]:
        """
        Returns functions to compute the heat fluxes of the thermal problem,
        representing convective and radiative effects, using Neumann boundary conditions

        :returns: Functions for exterior and interior surface heat fluxes.
        """

        def exterior_facade_heat_flux(
            current_temperature: jnp.ndarray,
            point: jnp.ndarray,
            old_temperature: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Calculates the total heat flux on the exterior surface of the facade,
            accounting for forced convection, solar radiation gain, and radiative heat
            loss.

            :returns: The total heat flux on the exterior surface
            """
            heat_flux_forced_convection = self.h_forced * (
                self.exterior_temperature - current_temperature[0]
            )

            solar_radiation_gain = self.absorptivity * self.solar_irradiance
            radiative_heat_loss = (
                self.SB_constant
                * self.emissivity
                * (current_temperature[0] ** 4 - self.exterior_temperature**4)
            )
            heat_flux_radiation = solar_radiation_gain - radiative_heat_loss

            total_heat_flux = heat_flux_forced_convection + heat_flux_radiation
            return -jnp.array([total_heat_flux])

        def interior_facade_heat_flux(
            current_temperature: jnp.ndarray, point: jnp.ndarray, old_T: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Calculates the total heat flux on the interior surface of the facade,
            accounting for natural convection.

            :returns: The total heat flux on the inerior surface
            """
            heat_flux_natural_convection = self.h_natural * (
                self.interior_temperature - current_temperature[0]
            )
            total_heat_flux = heat_flux_natural_convection
            return -jnp.array([total_heat_flux])

        return exterior_facade_heat_flux, interior_facade_heat_flux

    def set_params(self, params: list[jnp.ndarray]) -> None:
        """
        Updates internal temperature variables for the simulation with `params`, the
        previous time step temperature data. Converts FEM degrees of freedom (DOF) into
        temperature values for the domain's interior and boundary surfaces.
        """
        old_temperature_solution = params[0]
        old_temperature_solution_exterior = self.fes[0].convert_from_dof_to_face_quad(
            old_temperature_solution, self.boundary_inds_list[0]
        )
        old_temperature_solution_interior = self.fes[0].convert_from_dof_to_face_quad(
            old_temperature_solution, self.boundary_inds_list[1]
        )

        self.internal_vars = [
            self.fes[0].convert_from_dof_to_quad(old_temperature_solution)
        ]
        self.internal_vars_surfaces = [
            [old_temperature_solution_exterior],
            [old_temperature_solution_interior],
        ]

    @staticmethod
    def wind_to_h(wind_speed: float) -> float:
        """In forced convection case, calculate the convection heat transfer coefficient
        from the `wind_speed`

        :returns: The forced convection heat transfer coefficient.
        :raises valueError: raises when `wind_speed` is less than 0
        """
        if wind_speed < 0:
            raise ValueError("wind speed cannot be less than zero")
        return 18.6 * (wind_speed**0.605)

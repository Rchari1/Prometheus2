# coding=utf-8
"""
This file stores various functions related
to the properties of the gas (e.g. number densities, velocities,
absorption cross sections).
Created on 19. October 2021 by Andrea Gebek.
"""

import numpy as np
from scipy.special import erf, wofz
import os
import h5py
from scipy.interpolate import interp1d, RegularGridInterpolator
from . import constants as const
from . import geometryHandler as geom
from copy import deepcopy
from scipy.ndimage import gaussian_filter as gauss

lineListPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Resources/LineList.txt'
molecularLookupPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/molecularResources/'

class CollisionalAtmosphere:
    def __init__(self, T, P_0):
        self.T = T
        self.P_0 = P_0
        self.constituents = []
        self.hasMoon = False

    def getReferenceNumberDensity(self):
        n_0 = self.P_0 / (const.k_B * self.T)
        return n_0

    def getVelDispersion(self, m):
        sigma_v = np.sqrt(self.T * const.k_B / m)
        return sigma_v

    def addConstituent(self, speciesName, chi):
        species = const.AvailableSpecies().findSpecies(speciesName)
        m = species.mass
        sigma_v = self.getVelDispersion(m)
        constituent = AtmosphericConstituent(species, chi, sigma_v)
        self.constituents.append(constituent)

    def addMolecularConstituent(self, speciesName, chi):
        constituent = MolecularConstituent(speciesName, chi)
        self.constituents.append(constituent)

class BarometricAtmosphere(CollisionalAtmosphere):
    def __init__(self, T, P_0, mu, planet):
        super().__init__(T, P_0)
        self.mu = mu
        self.planet = planet

    def calculateNumberDensity(self, x, phi, rho, orbphase):
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = self.getReferenceNumberDensity()
        H = const.k_B * self.T * self.planet.R**2 / (const.G * self.mu * self.planet.M)
        n = n_0 * np.exp((self.planet.R - r) / H) * np.heaviside(r - self.planet.R, 1.)
        return n

class HydrostaticAtmosphere(CollisionalAtmosphere):
    def __init__(self, T, P_0, mu, planet):
        super().__init__(T, P_0)
        self.mu = mu
        self.planet = planet

    def calculateNumberDensity(self, x, phi, rho, orbphase):
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = self.getReferenceNumberDensity()
        Jeans_0 = const.G * self.mu * self.planet.M / (const.k_B * self.T * self.planet.R)
        Jeans = const.G * self.mu * self.planet.M / (const.k_B * self.T * r) * np.heaviside(r - self.planet.R, 1.)
        n = n_0 * np.exp(Jeans - Jeans_0)
        return n

class PowerLawAtmosphere(CollisionalAtmosphere):
    def __init__(self, T, P_0, q, planet):
        super().__init__(T, P_0)
        self.q = q
        self.planet = planet

    def calculateNumberDensity(self, x, phi, rho, orbphase):
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = self.getReferenceNumberDensity()
        n = n_0 * (self.planet.R / r)**self.q * np.heaviside(r - self.planet.R, 1.)
        return n

class EvaporativeExosphere:
    def __init__(self, N):
        self.N = N
        self.hasMoon = False

    def addConstituent(self, speciesName, sigma_v):
        species = const.AvailableSpecies().findSpecies(speciesName)
        constituent = AtmosphericConstituent(species, 1., sigma_v)
        self.constituents = [constituent]

    def addMolecularConstituent(self, speciesName, T):
        constituent = MolecularConstituent(speciesName, 1.0)
        self.constituents = [constituent]
        self.T = T

class PowerLawExosphere(EvaporativeExosphere):
    def __init__(self, N, q, planet):
        super().__init__(N)
        self.q = q
        self.planet = planet

    def calculateNumberDensity(self, x, phi, rho, orbphase):
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.planet.R**3) * self.N
        n = n_0 * (self.planet.R / r)**self.q * np.heaviside(r - self.planet.R, 1.)
        return n

class MoonExosphere(EvaporativeExosphere):
    def __init__(self, N, q, moon):
        super().__init__(N)
        self.q = q
        self.moon = moon
        self.hasMoon = True
        self.planet = moon.hostPlanet

    def calculateNumberDensity(self, x, phi, rho, orbphase):
        r = self.moon.getDistanceFromMoon(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.moon.R**3) * self.N
        n = n_0 * (self.moon.R / r)**self.q * np.heaviside(r - self.moon.R, 1.)
        return n
    
class TidallyHeatedMoon(EvaporativeExosphere):
    def __init__(self, q, moon):
        self.q = q
        self.moon = moon
        self.hasMoon = True
        self.planet = moon.hostPlanet

    def addSourceRateFunction(self, filename, tau_photoionization, mass_absorber):
        Mdot = np.loadtxt(filename)
        Mdot = np.concatenate((Mdot, Mdot[::-1]))
        phi_moon = np.linspace(0., 2. * np.pi, len(Mdot))
        N_function = interp1d(phi_moon, np.log10(Mdot * tau_photoionization / mass_absorber))
        self.N_function = N_function

    def calculateAbsorberNumber(self, orbphase):
        orbphase_moon = self.moon.getOrbphase(orbphase) % (2. * np.pi)
        N = 10**self.N_function(orbphase_moon)
        return N

    def calculateNumberDensity(self, x, phi, rho, orbphase):
        N = self.calculateAbsorberNumber(orbphase)
        r = self.moon.getDistanceFromMoon(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.moon.R**3) * N
        n = n_0 * (self.moon.R / r)**self.q * np.heaviside(r - self.moon.R, 1.)
        return n

class TorusExosphere(EvaporativeExosphere):
    def __init__(self, N, a_torus, v_ej, planet):
        super().__init__(N)
        self.a_torus = a_torus
        self.v_ej = v_ej
        self.planet = planet

    def calculateNumberDensity(self, x, phi, rho, orbphase):
        a, z = self.planet.getTorusCoords(x, phi, rho, orbphase)
        v_orbit = np.sqrt(const.G * self.planet.M / self.a_torus)
        H_torus = self.a_torus * self.v_ej / v_orbit
        n_a = np.exp(-((a - self.a_torus) / (4. * H_torus))**2)
        n_z = np.exp(-(z / H_torus)**2)
        term1 = 8. * H_torus**2 * np.exp(-self.a_torus**2 / (16. * H_torus**2))
        term2 = 2. * np.sqrt(np.pi) * self.a_torus * H_torus * (erf(self.a_torus / (4. * H_torus)) + 1.)
        n_0 = 1. / (2. * np.pi**1.5 * H_torus * (term1 + term2)) * self.N
        n = n_0 * np.multiply(n_a, n_z)
        return n

class SerpensExosphere(EvaporativeExosphere):
    def __init__(self, filename, N, planet, sigmaSmoothing):
        super().__init__(N)
        self.filename = filename
        self.planet = planet
        self.sigmaSmoothing = sigmaSmoothing

    def addInterpolatedDensity(self, spatialGrid):
        serpensOutput = np.loadtxt(self.filename) * 1e2
        particlePos = serpensOutput[:, 0:3]
        xBins = spatialGrid.constructXaxis(midpoints = False)
        yBins = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps) + 1)
        zBins = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps) + 1)
        cellVolume = (xBins[1] - xBins[0]) * (yBins[1] - yBins[0]) * (zBins[1] - zBins[0])
        n_histogram = np.histogramdd(particlePos, bins = [xBins, yBins, zBins])[0] * self.N / (np.size(particlePos, axis = 0) * cellVolume)
        if self.sigmaSmoothing > 0.:
            n_histogram = gauss(n_histogram, sigma = self.sigmaSmoothing)
        print('Sum over all particles, potentially smoothed with a Gaussian:', np.sum(n_histogram) * cellVolume)
        xPoints = spatialGrid.constructXaxis()
        yPoints = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps), endpoint = False) + 2. * spatialGrid.rho_border / (4. * spatialGrid.rho_steps)
        zPoints = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps), endpoint = False) + 2. * spatialGrid.rho_border / (4. * spatialGrid.rho_steps)
        x, y, z = np.meshgrid(xPoints, yPoints, zPoints, indexing = 'ij')
        SEL = ((y**2 + z**2) > self.planet.R**2) * ((y**2 + z**2) < self.planet.hostStar.R**2)
        print('Sum over all particles outside of the planetary disk but inside the stellar disk:', np.sum(n_histogram[SEL]) * cellVolume)
        n_function = RegularGridInterpolator((xPoints, yPoints, zPoints), n_histogram)
        self.InterpolatedDensity = n_function

    def calculateNumberDensity(self, x, phi, rho, orbphase):
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        coordArray = np.array([x, np.repeat(y, np.size(x)), np.repeat(z, np.size(x))]).T
        n = self.InterpolatedDensity(coordArray)
        return n

"""
Calculate absorption cross sections
"""

class AtmosphericConstituent:
    def __init__(self, species, chi, sigma_v):
        self.isMolecule = False
        self.species = species
        self.chi = chi
        self.sigma_v = sigma_v
        self.wavelengthGridRefinement = 10.
        self.wavelengthGridExtension = 0.01
        self.lookupOffset = 1e-50

    def getLineParameters(self, wavelength):
        lineList = np.loadtxt(lineListPath, dtype = str, usecols = (0, 1, 2, 3, 4), skiprows = 1)
        line_wavelength = np.array([x[1:-1] for x in lineList[:, 2]])
        line_A = np.array([x[1:-1] for x in lineList[:, 3]])
        line_f = np.array([x[1:-1] for x in lineList[:, 4]])
        
        SEL_COMPLETE = (line_wavelength != '') * (line_A != '') * (line_f != '')
        SEL_SPECIES = (lineList[:, 0] == self.species.element) * (lineList[:, 1] == self.species.ionizationState)
        line_wavelength = line_wavelength[SEL_SPECIES * SEL_COMPLETE].astype(float) * 1e-8
        line_gamma = line_A[SEL_SPECIES * SEL_COMPLETE].astype(float) / (4. * np.pi)
        line_f = line_f[SEL_SPECIES * SEL_COMPLETE].astype(float)
        SEL_WAVELENGTH = (line_wavelength > min(wavelength)) * (line_wavelength < max(wavelength))
        return line_wavelength[SEL_WAVELENGTH], line_gamma[SEL_WAVELENGTH], line_f[SEL_WAVELENGTH]

    def calculateVoigtProfile(self, wavelength):
        line_wavelength, line_gamma, line_f = self.getLineParameters(wavelength)
        sigma_abs = np.zeros_like(wavelength)
        for idx in range(len(line_wavelength)):
            lineProfile = wofz((const.c / wavelength - const.c / line_wavelength[idx] + 1j * line_gamma[idx]) / (self.sigma_v / line_wavelength[idx] * np.sqrt(2))).real
            lineProfile /= (self.sigma_v / line_wavelength[idx] * np.sqrt(2 * np.pi))
            sigma_abs += np.pi * (const.e)**2 / (const.m_e * const.c) * line_f[idx] * lineProfile
        return sigma_abs

    def constructLookupFunction(self, wavelengthGrid):
        wavelengthGridRefined = deepcopy(wavelengthGrid)
        wavelengthGridRefined.resolutionHigh /= self.wavelengthGridRefinement
        wavelengthGridRefined.lower_w *= (1. - self.wavelengthGridExtension)
        wavelengthGridRefined.upper_w *= (1. + self.wavelengthGridExtension)
        wavelengthRefined = wavelengthGridRefined.constructWavelengthGridSingle(self)
        sigma_abs = self.calculateVoigtProfile(wavelengthRefined)
        lookupFunction = interp1d(wavelengthRefined, np.log10(sigma_abs + self.lookupOffset), bounds_error = False, fill_value = np.log10(self.lookupOffset))
        return lookupFunction

    def addLookupFunctionToConstituent(self, wavelengthGrid):
        lookupFunction = self.constructLookupFunction(wavelengthGrid)
        self.lookupFunction = lookupFunction

    def getSigmaAbs(self, wavelength):
        sigma_absFlattened = 10**self.lookupFunction(wavelength.flatten()) - self.lookupOffset
        sigma_abs = sigma_absFlattened.reshape(wavelength.shape)
        return sigma_abs

class MolecularConstituent:
    def __init__(self, moleculeName, chi):
        self.isMolecule = True
        self.lookupOffset = 1e-50
        self.moleculeName = moleculeName
        self.chi = chi

    def constructLookupFunction(self):
        with h5py.File(molecularLookupPath + self.moleculeName + '.h5', 'r+') as f:
            P = f['p'][:] * 10.
            T = f['t'][:]
            wavelength = 1. / f['bin_edges'][:][::-1]
            sigma_abs = f['xsecarr'][:][:, :, ::-1]
            lookupFunction = RegularGridInterpolator((P, T, wavelength), np.log10(sigma_abs + self.lookupOffset), bounds_error = False, fill_value = np.log10(self.lookupOffset))
            return lookupFunction

    def addLookupFunctionToConstituent(self):
        lookupFunction = self.constructLookupFunction()
        self.lookupFunction = lookupFunction

    def getSigmaAbs(self, P, T, wavelength):
        wavelengthFlattened = wavelength.flatten()
        TFlattened = np.full_like(wavelengthFlattened, T)
        PFlattened = np.repeat(np.clip(P, a_min = 1e-4, a_max = None), np.size(wavelength, axis = 1))
        inputArray = np.array([PFlattened, TFlattened, wavelengthFlattened]).T
        sigma_absFlattened = 10**self.lookupFunction(inputArray) - self.lookupOffset
        sigma_abs = sigma_absFlattened.reshape(wavelength.shape)
        return sigma_abs

class Atmosphere:
    def __init__(self, densityDistributionList, hasOrbitalDopplerShift):
        self.densityDistributionList = densityDistributionList
        self.hasOrbitalDopplerShift = hasOrbitalDopplerShift

    def getAbsorberNumberDensity(self, densityDistribution, chi, x, phi, rho, orbphase):
        n_total = densityDistribution.calculateNumberDensity(x, phi, rho, orbphase)
        n_abs = n_total * chi
        return n_abs

    def getAbsorberVelocityField(self, densityDistribution, x, phi, rho, orbphase):
        v_los = np.zeros_like(x)
        if self.hasOrbitalDopplerShift:
            if not densityDistribution.hasMoon:
                v_los += densityDistribution.planet.getLOSvelocity(orbphase)
            else:
                v_los += densityDistribution.moon.getLOSvelocity(orbphase)
        return v_los

    def getLOSopticalDepth(self, x, phi, rho, orbphase, wavelength, delta_x):
        kappa = np.zeros((len(x), len(wavelength)))
        for densityDistribution in self.densityDistributionList:
            for constituent in densityDistribution.constituents:
                v_los = self.getAbsorberVelocityField(densityDistribution, x, phi, rho, orbphase)
                shift = const.calculateDopplerShift(-v_los)
                wavelengthShifted = np.tensordot(shift, wavelength, axes = 0)
                if constituent.isMolecule:
                    n_tot = densityDistribution.calculateNumberDensity(x, phi, rho, orbphase)
                    n_abs = n_tot * constituent.chi
                    T = densityDistribution.T
                    P = n_tot * const.k_B * T
                    sigma_abs = constituent.getSigmaAbs(P, T, wavelengthShifted)
                else:
                    n_abs = self.getAbsorberNumberDensity(densityDistribution, constituent.chi, x, phi, rho, orbphase)
                    sigma_abs = constituent.getSigmaAbs(wavelengthShifted)
                kappa += np.tile(n_abs, (len(wavelength), 1)).T * sigma_abs
        LOStau = np.sum(kappa, axis = 0) * delta_x
        return LOStau

class WavelengthGrid:
    def __init__(self, lower_w, upper_w, widthHighRes, resolutionLow, resolutionHigh):
        self.lower_w = lower_w
        self.upper_w = upper_w
        self.widthHighRes = widthHighRes
        self.resolutionLow = resolutionLow
        self.resolutionHigh = resolutionHigh

    def arangeWavelengthGrid(self, linesList):
        peaks = np.sort(np.unique(linesList))
        diff = np.concatenate(([np.inf], np.diff(peaks), [np.inf]))
        if len(peaks) == 0:
            print('WARNING: No absorption lines from atoms/ions in the specified wavelength range!')
            return np.arange(self.lower_w, self.upper_w, self.resolutionLow)
        HighResBorders = ([], [])
        for idx, peak in enumerate(peaks):
            if diff[idx] > self.widthHighRes:
                HighResBorders[0].append(peak - self.widthHighRes / 2.)
            if diff[idx + 1] > self.widthHighRes:
                HighResBorders[1].append(peak + self.widthHighRes / 2.)
        grid = []
        for idx in range(len(HighResBorders[0])):
            grid.append(np.arange(HighResBorders[0][idx], HighResBorders[1][idx], self.resolutionHigh))
            if idx == 0:
                if self.lower_w < HighResBorders[0][0]:
                    grid.append(np.arange(self.lower_w, HighResBorders[0][0], self.resolutionLow))
                if len(HighResBorders[0]) == 1 and self.upper_w > HighResBorders[1][-1]:
                    grid.append(np.arange(HighResBorders[1][0], self.upper_w, self.resolutionLow))
            elif idx == len(HighResBorders[0]) - 1:
                grid.append(np.arange(HighResBorders[1][idx - 1], HighResBorders[0][idx], self.resolutionLow))
                if self.upper_w > HighResBorders[1][-1]:
                    grid.append(np.arange(HighResBorders[1][-1], self.upper_w, self.resolutionLow))
            else:
                grid.append(np.arange(HighResBorders[1][idx - 1], HighResBorders[0][idx], self.resolutionLow))
        wavelengthGrid = np.sort(np.concatenate(grid))
        return wavelengthGrid

    def constructWavelengthGridSingle(self, constituent):
        linesList = constituent.getLineParameters(np.array([self.lower_w, self.upper_w]))[0]
        return self.arangeWavelengthGrid(linesList)

    def constructWavelengthGrid(self, densityDistributionList):
        linesList = []
        for densityDistribution in densityDistributionList:
            for constituent in densityDistribution.constituents:
                if constituent.isMolecule:
                    continue
                lines_w = constituent.getLineParameters(np.array([self.lower_w, self.upper_w]))[0]
                linesList.extend(lines_w)
        if len(linesList) == 0:
            return np.arange(self.lower_w, self.upper_w, self.resolutionLow)
        return self.arangeWavelengthGrid(linesList)

class Transit:
    def __init__(self, atmosphere, wavelengthGrid, spatialGrid):
        self.atmosphere = atmosphere
        self.wavelengthGrid = wavelengthGrid
        self.spatialGrid = spatialGrid
        self.planet = self.atmosphere.densityDistributionList[0].planet

    def addWavelength(self):
        wavelength = self.wavelengthGrid.constructWavelengthGrid(self.atmosphere.densityDistributionList)
        self.wavelength = wavelength

    def checkBlock(self, phi, rho, orbphase):
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        y_p = self.planet.getPosition(orbphase)[1]
        blockingPlanet = (np.sqrt((y - y_p)**2 + z**2) < self.planet.R)
        if blockingPlanet:
            return True
        for densityDistribution in self.atmosphere.densityDistributionList:
            if densityDistribution.hasMoon:
                moon = densityDistribution.moon
                y_moon = moon.getPosition(orbphase)[1]
                blockingMoon = ((y - y_moon)**2 + z**2 < moon.R)
                if blockingMoon:
                    return True
        return False

    def evaluateChord(self, phi, rho, orbphase):
        Fstar = self.planet.hostStar.getFstar(phi, rho, self.wavelength)
        F_out = rho * Fstar * self.wavelength / self.wavelength
        if self.checkBlock(phi, rho, orbphase):
            F_in = np.zeros_like(self.wavelength)
            return F_in, F_out
        x = self.spatialGrid.constructXaxis()
        delta_x = self.spatialGrid.getDeltaX()
        tau = self.atmosphere.getLOSopticalDepth(x, phi, rho, orbphase, self.wavelength, delta_x)
        F_in = rho * Fstar * np.exp(-tau)
        return F_in, F_out

    def sumOverChords(self):
        chordGrid = self.spatialGrid.getChordGrid()
        F_in = []
        F_out = []
        for chord in chordGrid:
            Fsingle_in, Fsingle_out = self.evaluateChord(chord[0], chord[1], chord[2])
            F_in.append(Fsingle_in)
            F_out.append(Fsingle_out)
        F_in = np.array(F_in).reshape((self.spatialGrid.phi_steps * self.spatialGrid.rho_steps, self.spatialGrid.orbphase_steps, len(self.wavelength)))
        F_out = np.array(F_out).reshape((self.spatialGrid.phi_steps * self.spatialGrid.rho_steps, self.spatialGrid.orbphase_steps, len(self.wavelength)))
        R = np.sum(F_in, axis = 0) / np.sum(F_out, axis = 0)
        return R

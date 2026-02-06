class Materials:
    def __init__(self, name: str, dielectric_permitivity, density=None, Z=None, T = None, W_value = None):
        self.name = name
        self.density = density
        self.dielectric_permitivity = dielectric_permitivity
        self.atomic_number = Z
        self.temperature = T

    def __str__(self):
        return f"Materials : {self.name}; Dielectric Permitivity : {self.dielectric_permitivity}"
        

    
LiquidArgon = Materials(name="Liquid Argon",
                        T=87, # K
                        Z=18,
                        dielectric_permitivity=1.62,
                        density=1.397,                 # g/cm³ http://webbook.nist.gov/chemistry/fluid/
                        W_value = 23.6
                        )

FR4 = Materials(name="FR4",
                dielectric_permitivity = 4.,
                density = 1.8 # https://fr.wikipedia.org/wiki/FR-4
                )

Copper = Materials(name = "Copper",
                   Z = 29,
                   dielectric_permitivity = 9999. #High dielectric permettivity to simulate a conductor
                   )
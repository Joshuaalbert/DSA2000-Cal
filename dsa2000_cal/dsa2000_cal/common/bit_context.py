import jax


class BitContext:

    def __init__(self, bits: int):
        if bits not in [32, 64]:
            raise ValueError("Bits must be either 32 or 64.")
        self.bits = bits
        self.previous_setting = jax.config.read("jax_enable_x64")

    def __enter__(self):
        if self.bits == 64:
            jax.config.update("jax_enable_x64", True)
        else:
            jax.config.update("jax_enable_x64", False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the previous setting
        jax.config.update("jax_enable_x64", self.previous_setting)


class BitContext32(BitContext):

    def __init__(self):
        super().__init__(32)


class BitContext64(BitContext):

    def __init__(self):
        super().__init__(64)

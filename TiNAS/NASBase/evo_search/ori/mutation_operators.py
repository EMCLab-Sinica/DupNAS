import copy
import random

from settings import Settings

# adapted from:
# https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm

class MutationOperator:
    def __init__(self, global_settings: Settings, arch_manager, num_blocks, **kwargs):
        self.num_blocks = num_blocks
        self.arch_manager = arch_manager

        self.mutate_prob = global_settings.NAS_EVOSEARCH_SETTINGS['MUT_PROB']
        self.mutate_prob_per_block = global_settings.TINAS['STAGE2_SETTINGS']['MUT_PROB_PER_BLOCK']

        # for mutate_adaptive operator
        self.mutate_prob_block_exploration = global_settings.TINAS['STAGE2_SETTINGS']['MUT_PROB_PER_BLOCK_EXPLORATION']
        self.mutate_prob_block_exploitation = global_settings.TINAS['STAGE2_SETTINGS']['MUT_PROB_PER_BLOCK_EXPLOITATION']
        self.best_stable_generations = global_settings.TINAS['STAGE2_SETTINGS']['BEST_STABLE_GENERATIONS']
        self.mutate_with_exploitation = False

    def mutate_default(self, sample):
        new_sample = copy.deepcopy(sample)

        # if random.random() < self.mutate_prob:
        #    self.arch_manager.random_resample_resolution(new_sample)

        for bix in range(self.num_blocks):
            if random.random() < self.mutate_prob:
                self.arch_manager.random_resample(new_sample, bix)

        # for i in range(self.num_stages):
        #     if random.random() < self.mutate_prob:
        #         self.arch_manager.random_resample_depth(new_sample, i)

        return new_sample

    def mutate_blockwise_prob(self, sample):
        new_sample = copy.deepcopy(sample)

        for bix in range(self.num_blocks):
            if random.random() < self.mutate_prob_per_block[bix]:
                msg = f'Mutate block {bix}: original={new_sample!r}, '

                self.arch_manager.random_resample(new_sample, bix)

                msg += f'mutated={new_sample!r}'
                print(msg)

        return new_sample

    # Check if there are N generations without improvements
    def _check_best_valids(self, best_valids):
        if len(best_valids) < self.best_stable_generations:
            return False

        # Check if best_valids[-1], best_valids[-2], ..., best_valids[-best_stable_generations] are all the same
        for idx in range(1, self.best_stable_generations + 1):
            if best_valids[-idx] != best_valids[-1]:
                return False

        return True

    def mutate_adaptive(self, sample, best_valids):
        if not self.mutate_with_exploitation:
            mutate_prob_per_block = self.mutate_prob_block_exploration
            if self._check_best_valids(best_valids):
                # If there are `best_stable` generations with the same score,
                # set a flag in this class to always use exploitation after that
                self.mutate_with_exploitation = True

        # Check the flag again, so that new probabilities will be used for the current generation
        if self.mutate_with_exploitation:
            mutate_prob_per_block = self.mutate_prob_block_exploitation

        print('Mutation probabilities:', mutate_prob_per_block)

        # Do mutation, similar to mutate_blockwise_prob operator
        new_sample = copy.deepcopy(sample)

        for bix in range(self.num_blocks):
            if random.random() < mutate_prob_per_block[bix]:
                msg = f'Mutate block {bix}: original={new_sample!r}, '

                self.arch_manager.random_resample(new_sample, bix)

                msg += f'mutated={new_sample!r}'
                print(msg)

        return new_sample

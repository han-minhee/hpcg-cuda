cudaError_t radix_sort_pairs(
    void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input,
    KeysOutputIterator keys_output, ValuesInputIterator values_input,
    ValuesOutputIterator values_output, unsigned int size,
    unsigned int begin_bit = 0, unsigned int end_bit = 8 * sizeof(Key),
    hipStream_t stream = 0, bool debug_synchronous = false) {
  bool ignored;
  return detail::radix_sort_impl<Config, false>(
      temporary_storage, storage_size, keys_input, nullptr, keys_output,
      values_input, nullptr, values_output, size, ignored, begin_bit, end_bit,
      stream, debug_synchronous);
}

inline  cudaError_t radix_sort_impl(
    void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input,
    typename std::iterator_traits<KeysInputIterator>::value_type *keys_tmp,
    KeysOutputIterator keys_output, ValuesInputIterator values_input,
    typename std::iterator_traits<ValuesInputIterator>::value_type *values_tmp,
    ValuesOutputIterator values_output, unsigned int size,
    bool &is_result_in_output, unsigned int begin_bit, unsigned int end_bit,
    hipStream_t stream, bool debug_synchronous) {
  using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
  using value_type =
      typename std::iterator_traits<ValuesInputIterator>::value_type;

  static_assert(
      std::is_same<key_type, typename std::iterator_traits<
                                 KeysOutputIterator>::value_type>::value,
      "KeysInputIterator and KeysOutputIterator must have the same value_type");
  static_assert(
      std::is_same<value_type, typename std::iterator_traits<
                                   ValuesOutputIterator>::value_type>::value,
      "ValuesInputIterator and ValuesOutputIterator must have the same "
      "value_type");

  using config = default_or_custom_config<
      Config,
      default_radix_sort_config<ROCPRIM_TARGET_ARCH, key_type, value_type>>;

  constexpr bool with_values =
      !std::is_same<value_type, ::rocprim::empty_type>::value;

  constexpr unsigned int max_radix_size = 1 << config::long_radix_bits;

  constexpr unsigned int scan_size =
      config::scan::block_size * config::scan::items_per_thread;
  constexpr unsigned int sort_size =
      config::sort::block_size * config::sort::items_per_thread;

  const unsigned int blocks =
      std::max(1u, ::rocprim::detail::ceiling_div(size, sort_size));
  const unsigned int blocks_per_full_batch =
      ::rocprim::detail::ceiling_div(blocks, scan_size);
  const unsigned int full_batches =
      blocks % scan_size != 0 ? blocks % scan_size : scan_size;
  const unsigned int batches =
      (blocks_per_full_batch == 1 ? full_batches : scan_size);
  const bool with_double_buffer = keys_tmp != nullptr;

  const unsigned int bits = end_bit - begin_bit;
  const unsigned int iterations =
      ::rocprim::detail::ceiling_div(bits, config::long_radix_bits);
  const unsigned int radix_bits_diff =
      config::long_radix_bits - config::short_radix_bits;
  const unsigned int short_iterations =
      radix_bits_diff != 0
          ? ::rocprim::min(iterations,
                           (config::long_radix_bits * iterations - bits) /
                               radix_bits_diff)
          : 0;
  const unsigned int long_iterations = iterations - short_iterations;

  const size_t batch_digit_counts_bytes = ::rocprim::detail::align_size(
      batches * max_radix_size * sizeof(unsigned int));
  const size_t digit_counts_bytes =
      ::rocprim::detail::align_size(max_radix_size * sizeof(unsigned int));
  const size_t keys_bytes =
      ::rocprim::detail::align_size(size * sizeof(key_type));
  const size_t values_bytes =
      with_values ? ::rocprim::detail::align_size(size * sizeof(value_type))
                  : 0;
  if (temporary_storage == nullptr) {
    storage_size = batch_digit_counts_bytes + digit_counts_bytes;
    if (!with_double_buffer) {
      storage_size += keys_bytes + values_bytes;
    }
    return hipSuccess;
  }

  if (debug_synchronous) {
    std::cout << "blocks " << blocks << '\n';
    std::cout << "blocks_per_full_batch " << blocks_per_full_batch << '\n';
    std::cout << "full_batches " << full_batches << '\n';
    std::cout << "batches " << batches << '\n';
    std::cout << "iterations " << iterations << '\n';
    std::cout << "long_iterations " << long_iterations << '\n';
    std::cout << "short_iterations " << short_iterations << '\n';
     cudaError_t error = hipStreamSynchronize(stream);
    if (error != hipSuccess)
      return error;
  }

  char *ptr = reinterpret_cast<char *>(temporary_storage);
  unsigned int *batch_digit_counts = reinterpret_cast<unsigned int *>(ptr);
  ptr += batch_digit_counts_bytes;
  unsigned int *digit_counts = reinterpret_cast<unsigned int *>(ptr);
  ptr += digit_counts_bytes;
  if (!with_double_buffer) {
    keys_tmp = reinterpret_cast<key_type *>(ptr);
    ptr += keys_bytes;
    values_tmp = with_values ? reinterpret_cast<value_type *>(ptr) : nullptr;
  }

  bool to_output = with_double_buffer || (iterations - 1) % 2 == 0;
  bool from_input = true;
  if (!with_double_buffer && to_output) {
    // Copy input keys and values if necessary (in-place sorting: input and
    // output iterators are equal)
    const bool keys_equal =
        ::rocprim::detail::are_iterators_equal(keys_input, keys_output);
    const bool values_equal =
        with_values &&
        ::rocprim::detail::are_iterators_equal(values_input, values_output);
    if (keys_equal || values_equal) {
       cudaError_t error = ::rocprim::transform(keys_input, keys_tmp, size,
                                              ::rocprim::identity<key_type>(),
                                              stream, debug_synchronous);
      if (error != hipSuccess)
        return error;

      if (with_values) {
         cudaError_t error = ::rocprim::transform(
            values_input, values_tmp, size, ::rocprim::identity<value_type>(),
            stream, debug_synchronous);
        if (error != hipSuccess)
          return error;
      }

      from_input = false;
    }
  }

  unsigned int bit = begin_bit;
  for (unsigned int i = 0; i < long_iterations; i++) {
     cudaError_t error =
        radix_sort_iteration<config, config::long_radix_bits, Descending>(
            keys_input, keys_tmp, keys_output, values_input, values_tmp,
            values_output, size, batch_digit_counts, digit_counts, from_input,
            to_output, bit, end_bit, blocks_per_full_batch, full_batches,
            batches, stream, debug_synchronous);
    if (error != hipSuccess)
      return error;

    is_result_in_output = to_output;
    from_input = false;
    to_output = !to_output;
    bit += config::long_radix_bits;
  }
  for (unsigned int i = 0; i < short_iterations; i++) {
     cudaError_t error =
        radix_sort_iteration<config, config::short_radix_bits, Descending>(
            keys_input, keys_tmp, keys_output, values_input, values_tmp,
            values_output, size, batch_digit_counts, digit_counts, from_input,
            to_output, bit, end_bit, blocks_per_full_batch, full_batches,
            batches, stream, debug_synchronous);
    if (error != hipSuccess)
      return error;

    is_result_in_output = to_output;
    from_input = false;
    to_output = !to_output;
    bit += config::short_radix_bits;
  }

  return hipSuccess;
}
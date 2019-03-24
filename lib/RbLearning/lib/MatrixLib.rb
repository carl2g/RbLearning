require 'ffi'

module MatrixLib
	extend FFI::Library
	ffi_lib '/usr/lib/RbLearning/c_lib.so'
	attach_function :dot ,[:pointer, :pointer, :int, :int, :int], :pointer
	attach_function :mult ,[:pointer, :pointer, :int], :pointer
	attach_function :subtract ,[:pointer, :pointer, :int], :pointer
	attach_function :transpose ,[:pointer, :int, :int], :pointer
	attach_function :add ,[:pointer, :pointer, :int, :int], :pointer
end

module CMath
	extend FFI::Library
	ffi_lib 'm'
  	attach_function :exp, [:double], :double
  	attach_function :tanh, [:double], :double
end

module LibC
	extend FFI::Library
	ffi_lib FFI::Library::LIBC
	attach_function :malloc, [:size_t], :pointer
  	attach_function :free, [:pointer], :void
end

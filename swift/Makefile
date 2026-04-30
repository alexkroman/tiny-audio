# Makefile for TinyAudio Swift package
#
# On macOS with only Command Line Tools (no full Xcode), the Swift Testing
# framework lives in a developer framework path that SwiftPM does not search
# by default.  We detect that path at make-time and pass the needed -F flags
# to swift test so both compilation and the dyld runtime loader can find it.

TESTING_FRAMEWORK_PATH := $(shell \
  if [ -d "/Applications/Xcode.app/Contents/Developer/Library/Frameworks" ]; then \
    echo "/Applications/Xcode.app/Contents/Developer/Library/Frameworks"; \
  elif [ -d "/Library/Developer/CommandLineTools/Library/Developer/Frameworks" ]; then \
    echo "/Library/Developer/CommandLineTools/Library/Developer/Frameworks"; \
  fi)

SWIFT_TEST_FLAGS :=
ifneq ($(TESTING_FRAMEWORK_PATH),)
  SWIFT_TEST_FLAGS := \
    -Xswiftc -F -Xswiftc "$(TESTING_FRAMEWORK_PATH)" \
    -Xlinker -F -Xlinker "$(TESTING_FRAMEWORK_PATH)" \
    -Xlinker -rpath -Xlinker "$(TESTING_FRAMEWORK_PATH)"
endif

.PHONY: test test-filter build

build:
	swift build

test:
	swift test $(SWIFT_TEST_FLAGS)

test-filter:
	swift test $(SWIFT_TEST_FLAGS) --filter $(FILTER)

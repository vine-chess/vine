#include "options.hpp"
#include <iostream>
#include <memory>
#include <sstream>

namespace uci {

std::ostream &operator<<(std::ostream &os, const Option &opt) {
    opt.print(os);
    return os;
}

std::string Option::name() const {
    return name_;
}

IntegerOption::IntegerOption(std::string_view name, i32 value, i32 min, i32 max,
                             std::function<void(const Option &)> callback)
    : value_(value), min_(min), max_(max) {
    name_ = name;
    callback_ = std::move(callback);
    callback_(*this);
}

void IntegerOption::set_value(std::string_view str_value) {
    std::istringstream ss((std::string(str_value)));
    i32 new_value;
    if (ss >> new_value && new_value >= min_ && new_value <= max_) {
        value_ = new_value;
        callback_(*this);
    } else {
        std::cerr << "IntegerOption::set_value: invalid value '" << str_value << "' (expected " << min_ << " to " << max_
                  << ")" << std::endl;
    }
}

[[nodiscard]] std::string IntegerOption::value() const {
    return std::to_string(value_);
}

[[nodiscard]] std::string_view IntegerOption::type() const {
    return "spin";
}

void IntegerOption::print(std::ostream &os) const {
    os << "option name " << name_ << " type spin"
       << " default " << value_ << " min " << min_ << " max " << max_;
}

BoolOption::BoolOption(std::string_view name, bool value, std::function<void(const Option &)> callback)
    : value_(value) {
    name_ = name;
    callback_ = std::move(callback);
    callback_(*this);
}

void BoolOption::set_value(std::string_view str_value) {
    std::string lower(str_value);
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "true") {
        value_ = true;
    } else if (lower == "false") {
        value_ = false;
    } else {
        std::cerr << "BoolOption::set_value: invalid value '" << str_value << "'\n";
        return;
    }

    callback_(*this);
}

std::string BoolOption::value() const {
    return value_ ? "true" : "false";
}

std::string_view BoolOption::type() const {
    return "check";
}

void BoolOption::print(std::ostream &os) const {
    os << "option name " << name_ << " type check"
       << " default " << value_;
}

StringOption::StringOption(std::string_view name, std::string value, std::function<void(const Option &)> callback)
    : value_(std::move(value)) {
    name_ = name;
    callback_ = std::move(callback);
    callback_(*this);
}

void StringOption::set_value(std::string_view str_value) {
    value_ = std::string(str_value);
    callback_(*this);
}

std::string StringOption::value() const {
    return std::string(value_);
}

std::string_view StringOption::type() const {
    return "string";
}

void StringOption::print(std::ostream &os) const {
    os << "option name " << name_ << " type string"
       << " default " << value_;
}

void Options::add(std::unique_ptr<Option> option) {
    map_.emplace(option->name(), std::move(option));
}

std::unique_ptr<Option> &Options::get(std::string_view name) {
    auto it = map_.find(std::string(name));
    if (it != map_.end()) {
        return it->second;
    } else {
        throw std::runtime_error("Options::get: option '" + std::string(name) + "' not found");
    }
}

std::ostream &operator<<(std::ostream &os, const Options &options) {
    for (const auto &[_, opt] : options.map_) {
        os << *opt << std::endl;
    }
    return os;
}

} // namespace uci
